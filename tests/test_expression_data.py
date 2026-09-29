import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data.expression_dataset import ExpressionDataset, collate, image_to_tensor
from src.data.expression_images import TARGET_GLYPH_EXTENT, estimate_glyph_extent, normalize_expression_image
from src.data.inkml import parse_inkml, render_traces
from src.data.latex_tokenizer import canonicalize, in_vocab, structure_type
from src.data.synthetic_expressions import CLASS_TO_LATEX, Grammar, Renderer, to_tokens


def _three_glyph_canvas() -> Image.Image:
    img = Image.new("L", (600, 200), 255)
    d = ImageDraw.Draw(img)
    for x in (50, 200, 350):
        d.rectangle([x, 50, x + 60, 150], outline=0, width=10)  # 100px-tall "glyphs"
    return img


def test_normalize_blank_returns_none():
    assert normalize_expression_image(Image.new("L", (100, 100), 255)) is None


def test_normalize_rescales_median_glyph_to_target_size():
    out = normalize_expression_image(_three_glyph_canvas())
    extent = estimate_glyph_extent(np.array(out))
    assert abs(extent - TARGET_GLYPH_EXTENT) <= 2


def test_dataset_and_serving_use_the_same_preprocessing(tmp_path):
    # Train/serve consistency: the Dataset's tensor for an image must equal
    # what serving (image_to_tensor) produces for the same image.
    _three_glyph_canvas().save(tmp_path / "e.png")
    ds = ExpressionDataset([{"image": "e.png", "tokens": ["a"]}], tmp_path, augment=False)
    served = image_to_tensor(Image.open(tmp_path / "e.png"))
    assert torch.equal(ds[0][0], served)


def test_collate_pads_to_shape_buckets_with_mask():
    from src.data.expression_dataset import H_BUCKET, T_BUCKET, W_BUCKET

    a = (torch.zeros(1, 10, 30), torch.tensor([1, 2, 3]))
    b = (torch.zeros(1, 13, 21), torch.tensor([1, 2]))
    images, mask, tgt_in, tgt_out = collate([a, b])
    # Bucketed shapes (regression: per-shape MPS caching leaked memory).
    assert images.shape[2] % H_BUCKET == 0 and images.shape[3] % W_BUCKET == 0
    assert images.shape[2] % 4 == 0 and images.shape[3] % 4 == 0
    assert mask[0, :10, :30].all() and not mask[0, 10:].any()
    assert tgt_in.shape == tgt_out.shape and tgt_in.shape[1] % T_BUCKET == 0
    assert tgt_in[0, :2].tolist() == [1, 2] and tgt_out[0, :2].tolist() == [2, 3]


def test_bucket_sampler_covers_every_index_once_with_size_matched_batches():
    from src.data.expression_dataset import BucketBatchSampler

    sizes = [(10, w) for w in random.Random(0).sample(range(10, 1000), 200)]
    batches = list(BucketBatchSampler(sizes, batch_size=8, chunk_batches=25))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(200))
    # within a batch widths are contiguous in sorted order -> small spread
    spreads = [max(sizes[i][1] for i in b) - min(sizes[i][1] for i in b) for b in batches]
    assert np.median(spreads) < 100


def test_bucket_sampler_caps_padded_pixels_per_batch():
    # Regression test: fixed-size batches of the largest images used 9-12 GB
    # of MPS (= system) memory. Every batch must respect the pixel budget.
    from src.data.expression_dataset import BucketBatchSampler

    rng = random.Random(1)
    sizes = [(rng.randint(20, 136), rng.randint(40, 520)) for _ in range(500)]
    sampler = BucketBatchSampler(sizes, batch_size=32, max_pixels=100_000, chunk_batches=10)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(500))
    for b in batches:
        assert len(b) == 1 or len(b) * max(sizes[i][0] for i in b) * max(sizes[i][1] for i in b) <= 100_000
    assert len(sampler) == len(batches)


def test_parse_inkml_handles_both_point_formats():
    mw = '<ink xmlns="http://www.w3.org/2003/InkML"><annotation type="normalizedLabel">x^{2}</annotation><trace>1 2 0, 3 4 10</trace></ink>'
    cr = '<ink xmlns="http://www.w3.org/2003/InkML"><annotation type="truth">$x^2$</annotation><trace>1 2, 3 4, 5 6</trace></ink>'
    traces, ann = parse_inkml(mw)
    assert ann["normalizedLabel"] == "x^{2}" and traces[0].shape == (2, 2)
    traces, ann = parse_inkml(cr)
    assert ann["truth"] == "$x^2$" and traces[0].shape == (3, 2)


def test_render_traces_draws_ink():
    img = render_traces([np.array([[0, 0], [10, 10]], dtype=np.float32), np.array([[20, 0], [20, 10]], dtype=np.float32)])
    assert img is not None and (np.array(img) < 128).any()
    assert render_traces([]) is None


def _fake_pools():
    glyph = np.full((20, 14), 255, dtype=np.uint8)
    glyph[2:18, 5:9] = 0
    return {cls: [glyph] for cls in CLASS_TO_LATEX}


GRAMMAR_CFG = {"max_top_level_terms": 3, "max_inner_terms": 2, "max_number_digits": 3, "max_depth": 3,
               "max_tokens": 40, "p_bracket_group": 0.08, "p_infty": 0.03}
LAYOUT_CFG = {"glyph_size": 32, "output_upscale": 3, "gap": [0.15, 0.55], "vertical_jitter": 0.12,
              "glyph_scale_jitter": [0.85, 1.15], "script_scale": [0.6, 0.8], "superscript_center": [-0.2, 0.3],
              "subscript_center": [-0.3, 0.2], "script_gap": [-0.08, 0.2], "fraction_part_scale": [0.75, 0.95],
              "fraction_bar_overhang": [1.05, 1.3], "fraction_gap": [0.08, 0.3]}


def test_synthetic_images_have_no_faint_phantom_components():
    # Regression test: resampling halos (gray pixels 200-249) used to form
    # detached phantom marks around strokes ("a" rendered as "a-"). Real ink
    # renders have none; neither should synthetic training images.
    import cv2

    from src.data.synthetic_expressions import generate

    cfg = {"grammar": GRAMMAR_CFG, "layout": LAYOUT_CFG,
           "structure_mix": {"flat": 0.25, "script": 0.25, "nested_script": 0.25, "fraction": 0.25}}
    for img, _tokens, _st in generate(_fake_pools(), cfg, 4, seed=0):
        a = np.array(img)
        assert not ((a >= 200) & (a < 250)).any()
        n_permissive = cv2.connectedComponents((a < 250).astype(np.uint8))[0]
        n_strict = cv2.connectedComponents((a < 200).astype(np.uint8))[0]
        assert n_permissive == n_strict


def test_synthetic_grammar_hits_every_requested_structure_type():
    rng = random.Random(0)
    grammar = Grammar(GRAMMAR_CFG, rng)
    renderer = Renderer(_fake_pools(), LAYOUT_CFG, rng)
    for target in ("flat", "script", "nested_script", "fraction"):
        tree = grammar.sample(target)
        tokens = canonicalize(to_tokens(tree))
        assert structure_type(tokens) == target
        assert in_vocab(tokens)
        box = renderer.render(tree, 32)
        assert (box.img < 128).any()
