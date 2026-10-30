# Third-party notices

The container image published at `forgejo.tail5d39b4.ts.net/jakub/social-gt` is built from this
repository's `Dockerfile` and redistributes the components below. This repository's own code is
MIT (see `LICENSE`); the components keep their own licenses.

## GNU GPL-2.0-or-later

| Component | Version | License | Source used by the build |
| --- | --- | --- | --- |
| Gambit / `pygambit` | 16.7.0 | GPL-2.0-or-later | PyPI sdist `pygambit-16.7.0.tar.gz`, compiled unmodified by the `Dockerfile` |

Gambit is licensed under the GNU General Public License, version 2 or (at your option) any later
version. The full license text is available at <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
and ships inside the sdist as `COPYING`.

Because the image is served over a network, the corresponding source for the GPL component is
available from the same place the build obtains it:

- PyPI: <https://pypi.org/project/pygambit/16.7.0/#files> (`pygambit-16.7.0.tar.gz`)
- Upstream: <https://github.com/gambitproject/gambit>

The sdist is built without source modification: `Dockerfile` runs `pip install` against it with the
pinned version from `requirements.txt`. No patches are applied, so the published sdist is the exact
corresponding source for the compiled component.

Written offer: the image publisher (Jakub Sikora) will supply the corresponding source archive on
request to any recipient of the image.

## Apache-2.0

| Component | Version | License | Source |
| --- | --- | --- | --- |
| OpenSpiel (`open_spiel`) | 2.0.2 | Apache-2.0 | <https://github.com/google-deepmind/open_spiel> |

## Permissive dependencies

`gradio`, `mcp`, `numpy`, `scipy`, `sympy`, `openai` and their transitive dependencies are
distributed under MIT, BSD, Apache-2.0 or PSF licenses; their license metadata is present in the
installed distributions inside the image (`*.dist-info/licenses`).