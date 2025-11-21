
import numpy as np
from brainspace.gradient.alignment import procrustes_alignment, ProcrustesAlignment

def test_alignment_options():
    rs = np.random.RandomState(0)
    # Create two datasets that are just shifted and scaled versions of each other
    d1 = rs.randn(10, 5)
    
    # Shift and scale d2
    shift = 5.0
    scale = 2.0
    d2 = d1 * scale + shift
    
    list_data = [d1, d2]

    # 1. Test with default options (center=False, scale=False)
    # Since d2 is shifted and scaled, they shouldn't align perfectly without centering and scaling
    aligned_default = procrustes_alignment(list_data, center=False, scale=False)
    
    # The distance between aligned datasets should be significant
    diff_default = np.linalg.norm(aligned_default[0] - aligned_default[1])
    assert diff_default > 1.0 # Arbitrary threshold, but should be large

    # 2. Test with center=True, scale=False
    # Should handle shift but not scale
    aligned_center = procrustes_alignment(list_data, center=True, scale=False)
    
    # If we only center, the scale difference remains. 
    # d2_centered = d2 - mean(d2) = (d1 * scale + shift) - (mean(d1) * scale + shift) = (d1 - mean(d1)) * scale
    # d1_centered = d1 - mean(d1)
    # So they are still different by a factor of scale.
    diff_center = np.linalg.norm(aligned_center[0] - aligned_center[1])
    assert diff_center > 0.1 # Still some difference due to scale

    # 3. Test with center=True, scale=True
    # Should align perfectly (or very close)
    aligned_both = procrustes_alignment(list_data, center=True, scale=True)
    
    diff_both = np.linalg.norm(aligned_both[0] - aligned_both[1])
    assert diff_both < 1e-5 # Should be very small

    # 4. Test using the class interface
    pa = ProcrustesAlignment(center=True, scale=True)
    pa.fit(list_data)
    
    diff_class = np.linalg.norm(pa.aligned_[0] - pa.aligned_[1])
    assert diff_class < 1e-5

if __name__ == "__main__":
    test_alignment_options()
