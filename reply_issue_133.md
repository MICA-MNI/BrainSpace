Hello @hasibagen,

Thank you for reporting this issue. You are correct that `read_surface` was not recognizing FreeSurfer surface file extensions like `.pial` and `.orig`, even though the underlying reader supports them.

We have updated `read_surface` to explicitly support common FreeSurfer geometry file extensions: `.pial`, `.white`, `.orig`, `.sphere`, `.inflated`, and `.smoothwm`. These files will now be automatically handled by the FreeSurfer reader.

You should now be able to read these files directly:

```python
from brainspace.mesh.mesh_io import read_surface
rh = read_surface('/surf/rh.pial')
```

This fix will be available in the next release.
