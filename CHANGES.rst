1.0.5 (2016-08-16)
------------------
- Updated to newest version of astropy package template.

- Fixed median cleaning. There was a subtle bug that the crmask was defined as a unit8
  array. This was then used to clean the image, but this acted as indexes 0 and 1 rather than
  a boolean array that was intended

1.0.4 (2016-02-29)
------------------

- Fixed setup_requires so that it doesn't install astropy when using egg_info.

- Pinned coverage version to 3.7.1.

- Removed dependence on endianness in tests

- Fixed build issues on windows


1.0.3 (2015-09-29)
------------------

- Updated URL in setup.cfg.

1.0.2 (2015-09-29)
------------------

- Added .h files to MANIFEST.in

1.0.1 (2015-09-29)
------------------

- Fixed bug in MANIFEST.in that was excluding *.pyx files.

1.0 (2015-09-29)
----------------

- Initial release.
