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


def test_nearby_similar_sized_strokes_get_merged_into_one_region():
    canvas = _blank_canvas()
    draw = ImageDraw.Draw(canvas)
    # Two short, similarly-sized strokes with a tiny gap -- like a single
    # symbol (e.g. '+' or 'x') drawn as two disconnected pen strokes --
    # should merge into one region.
    draw.rectangle([40, 40, 55, 90], fill=0)
    draw.rectangle([57, 40, 72, 90], fill=0)
    regions = detect_candidate_regions(canvas)
    assert len(regions) == 1


def test_tiny_noise_speck_is_filtered_out():
    canvas = _blank_canvas()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([40, 40, 80, 100], fill=0)  # real symbol
    draw.point((250, 120), fill=0)  # single stray pixel, far away
    regions = detect_candidate_regions(canvas, min_area=12)
    assert len(regions) == 1


def test_adjacent_digits_of_a_multi_digit_number_stay_separate():
    # Regression test: two digits written close together (as digits of one
    # multi-digit number naturally are) must NOT be merged into one
    # unclassifiable blob -- see bug report on '36' being detected as a
    # single object.
    canvas = _blank_canvas(size=(400, 200))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([60, 40, 95, 160], fill=0)  # '3'-ish block, height 120
    draw.rectangle([110, 40, 145, 160], fill=0)  # '6'-ish block, 15px gap
    regions = detect_candidate_regions(canvas)
    assert len(regions) == 2


def test_closely_drawn_superscript_stays_separate_from_its_base():
    # Regression test: a small, tightly-drawn superscript (completely normal
    # handwriting -- exponents are conventionally drawn close to their base)
    # must NOT be merged into the base, even with only a few pixels of gap,
    # because it's dramatically smaller -- that size disparity is exactly
    # what should signal "this is a modifier for Phase 4 to detect," not
    # "this is a disconnected stroke of the same symbol." See bug report on
    # x^2 being detected (and classified) as a single object.
    canvas = _blank_canvas(size=(400, 200))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([60, 60, 130, 160], fill=0)  # base, 70x100
    draw.rectangle([132, 50, 160, 85], fill=0)  # small, raised, 2px gap from base
    regions = detect_candidate_regions(canvas)
    assert len(regions) == 2
