from PIL import Image, ImageDraw

from src.recognition.components import detect_candidate_regions


def _blank_canvas(size=(300, 150)):
    return Image.new("L", size, color=255)


def test_no_ink_returns_no_regions():
    canvas = _blank_canvas()
    assert detect_candidate_regions(canvas) == []


def test_single_blob_returns_one_region():
    canvas = _blank_canvas()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([40, 40, 80, 100], fill=0)
    regions = detect_candidate_regions(canvas)
    assert len(regions) == 1
    assert regions[0].width > 0 and regions[0].height > 0


def test_two_well_separated_blobs_return_two_regions_in_reading_order():
    canvas = _blank_canvas()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([20, 40, 50, 100], fill=0)  # left blob
    draw.rectangle([200, 40, 230, 100], fill=0)  # right blob, far away
    regions = detect_candidate_regions(canvas)
    assert len(regions) == 2
    # sorted left-to-right by center
    assert regions[0].center[0] < regions[1].center[0]


def test_nearby_strokes_get_merged_into_one_region():
    canvas = _blank_canvas()
    draw = ImageDraw.Draw(canvas)
    # Two short strokes with a tiny gap -- like a single symbol drawn as two
    # disconnected pen strokes -- should merge into one region.
    draw.rectangle([40, 40, 55, 90], fill=0)
    draw.rectangle([57, 40, 72, 90], fill=0)
    regions = detect_candidate_regions(canvas, merge_gap_ratio=0.5)
    assert len(regions) == 1


def test_tiny_noise_speck_is_filtered_out():
    canvas = _blank_canvas()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([40, 40, 80, 100], fill=0)  # real symbol
    draw.point((250, 120), fill=0)  # single stray pixel, far away
    regions = detect_candidate_regions(canvas, min_area=12)
    assert len(regions) == 1
