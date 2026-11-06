# Third-party code

No third-party source code is redistributed in this repository. Each dependency is fetched
at a pinned commit and verified by SHA-256 before use.

| Component | Source | Pinned version | How it is obtained |
|---|---|---|---|
| GraphDPO (official code) | https://github.com/ljj-007/GraphDPO | `2b0aa9597fc0c9a4b122f21d2935265ad60da1f7` | `python code/scripts/fetch_upstream.py GraphDPO` |
| ZeroFisher official code (ZOWFKGIF) | https://github.com/NKUShaw/ZOWFKGIF | `9672e98756985576024818e8ac7bbee56f24bb89` | `python code/scripts/fetch_upstream.py ZOWFKGIF` |
| advertorch (used by the SSD/ADV-IMP baselines) | PyPI `advertorch==0.2.3` | 0.2.3 | `python code/scripts/fetch_upstream.py advertorch` |
| Schema-based unlearning | https://github.com/NKUShaw/KGUnlearningBySchema | `a938d254ee76a2e475004ce90356ab0db9541eed` | manual clone; see `schema_baseline/LICENSE_NOTICE.txt` |

The licenses of these projects apply to their code. Our own code is released under the MIT license (`LICENSE`).
