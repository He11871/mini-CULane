# mini-CULane image lists

These lists identify the images in the scene-balanced mini-CULane subset. Paths are relative to an existing CULane dataset root. Use the first field in each row to locate the image.

| Image list | Source | Unique images |
| --- | --- | ---: |
| [Training](list/train_gt.txt) | CULane training split | 10,000 |
| [Validation](list/val_gt.txt) | CULane validation split | 1,000 |
| [Test](list/test.txt) | Nine CULane test scenarios | 2,700 |

The sampling seed is **42**. The sampler shuffles the training and validation source lists separately. Within each split, it groups images by their top-level driver directory and takes one image at a time from each directory in sorted directory order until it reaches the target size. The three lists have no image paths in common.

For testing, the sampler initializes a random generator with seed 42, shuffles the nine official scenario lists in the order below, and takes the first 300 images from each. Each category accounts for one ninth of the test subset.

| Scenario | Scenario | Scenario |
| --- | --- | --- |
| 0 · [Normal](list/test_split/test0_normal.txt) | 1 · [Crowded](list/test_split/test1_crowd.txt) | 2 · [Dazzle light](list/test_split/test2_hlight.txt) |
| 3 · [Shadow](list/test_split/test3_shadow.txt) | 4 · [No line](list/test_split/test4_noline.txt) | 5 · [Arrow](list/test_split/test5_arrow.txt) |
| 6 · [Curve](list/test_split/test6_curve.txt) | 7 · [Crossroads](list/test_split/test7_cross.txt) | 8 · [Night](list/test_split/test8_night.txt) |

## Reproducing the selection

The [sampling script](scripts/get_miniculane.py) reads the original CULane lists. Set the source to your local CULane root and choose a new output directory.

```sh
python scripts/get_miniculane.py --src_root SOURCE_CULANE --dst_root NEW_MINI_ROOT --train_n 10000 --val_n 1000 --test_per_scene 300 --seed 42
```

The sampler writes image and mask paths for training and validation. The distributed lists also contain four lane-existence fields. These fields differ from the corresponding source fields on 6,158 training rows. Use the distributed lists to preserve their exact contents. The [manifest](manifest.json) gives the SHA-256 hashes of the 12 distributed lists and their source counterparts.

For a local subset copy, the [annotation helper](scripts/patch_miniculane.py) copies the lane polylines beside the images. Set its `SRC_ROOT` and `DST_ROOT` to your local directories before use. The [verification script](scripts/verify_miniculane.py) checks the subset files against the source and summarizes scenario counts. It requires NumPy and OpenCV.
