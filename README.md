# napari-metamorph

[![License BSD-3](https://img.shields.io/pypi/l/napari-metamorph.svg?color=green)](https://github.com/jdeseze/napari-metamorph/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-metamorph.svg?color=green)](https://pypi.org/project/napari-metamorph)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-metamorph.svg?color=green)](https://python.org)
[![tests](https://github.com/jdeseze/napari-metamorph/workflows/tests/badge.svg)](https://github.com/jdeseze/napari-metamorph/actions)
[![codecov](https://codecov.io/gh/jdeseze/napari-metamorph/branch/main/graph/badge.svg)](https://codecov.io/gh/jdeseze/napari-metamorph)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-metamorph)](https://napari-hub.org/plugins/napari-metamorph)

A simple plugin to read .nd files and .rgn files from Metamorph

It is inspired from the ImageJ plugin from Fabrice Cordelieres (https://imagej.net/ij/plugins/track/builder.html), many thanks to him! 

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-metamorph` via [pip]:

    pip install git+https://github.com/jdeseze/napari-metamorph.git

only basic modules are needed, but there are detailed in requirements.txt

## Usage

When installed, you can find the plugin in the 'Plugins' as 'Nd reader (napari-metamorph)'
When you click on it, a widget should appear on the right. 

![image](https://github.com/jdeseze/napari-metamorph/assets/68115566/aa21c612-736b-4bbd-bd7d-58e06ee74f8a)

With the 'Select folder' button, you can select folder in which there is at least one Metamorph experiment with an .nd file.

WARNING!! IT SHOULD BE THE RAW DATA, WITH ALL THE IMAGES TAKEN DURING A TIMELAPSE AS INDIVIDUAL FILES (AS THEY ARE SAVED BY METAMORPH)

The first dropdown list lets you choose the .nd file, if you have multiple experiments in this folder.

The second one lets you choose the stage position if you did multiple position. 

If you select the 'Keep layers' button, you  should load the new layers on top of the ones you already have. Otherwise, the new experiment will be loaded alone. 

You can click the 'Load image' button to see the images of the corresponding experiment+position. 

If you just want to go to the next position, you can click on the 'Next position' button. 

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

Don't hesitate to tell me if you want some easy features to be added. 

## License

Distributed under the terms of the [BSD-3] license,
"napari-metamorph" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

If it doesn't work, try to import the library in Pyhton. It happened that pooch was missiong, which only requires "conda -c conda-forge install pooch"

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/jdeseze/napari-metamorph/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
