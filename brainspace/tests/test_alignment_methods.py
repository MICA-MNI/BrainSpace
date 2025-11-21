
import numpy as np
from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import procrustes

def test_alignment_methods():
    rs = np.random.RandomState(0)
    # Create two random connectivity matrices
    n_nodes = 100
    c1 = rs.rand(n_nodes, n_nodes)
    c1 = (c1 + c1.T) / 2
    c2 = rs.rand(n_nodes, n_nodes)
    c2 = (c2 + c2.T) / 2
    c3 = rs.rand(n_nodes, n_nodes)
    c3 = (c3 + c3.T) / 2

    # Case 1: Joint alignment (GPA)
    gp = GradientMaps(kernel='normalized_angle', alignment='procrustes', n_components=3, random_state=0)
    gp.fit([c1, c2, c3])
    aligned_1 = gp.aligned_
    
    # Case 2: Align to reference (single matrix)
    # First compute gradients for c2 to use as reference
    gref = GradientMaps(kernel='normalized_angle', approach='dm', n_components=3, random_state=0)
    gref.fit(c2)
    ref_grads = gref.gradients_

    # Now align c1 to c2's gradients
    galign = GradientMaps(kernel='normalized_angle', alignment='procrustes', n_components=3, random_state=0)
    galign.fit(c1, reference=ref_grads)
    aligned_2 = galign.aligned_

    # Compare aligned_1[0] (c1 aligned in GPA) and aligned_2 (c1 aligned to c2)
    # They should be different because GPA aligns both to a mean, while Case 2 aligns c1 to c2.
    diff = np.linalg.norm(aligned_1[0] - aligned_2)
    assert diff > 0.1, "GPA and Fixed Alignment should produce different results"

    # Case 3: Align multiple to reference with n_iter=1 (Fixed Alignment for list)
    gp_fixed = GradientMaps(kernel='normalized_angle', alignment='procrustes', n_components=3, random_state=0)
    gp_fixed.fit([c1, c2, c3], reference=ref_grads, n_iter=1)
    aligned_3 = gp_fixed.aligned_
    
    # Check if aligned_3[0] is aligned to ref_grads
    aligned_check_3_0 = procrustes(gp_fixed.gradients_[0], ref_grads)
    diff_check_3_0 = np.linalg.norm(aligned_3[0] - aligned_check_3_0)
    assert diff_check_3_0 < 1e-5, "Case 3: c1 should be aligned to reference"
    
    # Check if aligned_3[1] is aligned to ref_grads
    aligned_check_3_1 = procrustes(gp_fixed.gradients_[1], ref_grads)
    diff_check_3_1 = np.linalg.norm(aligned_3[1] - aligned_check_3_1)
    assert diff_check_3_1 < 1e-5, "Case 3: c2 should be aligned to reference"

    # Case 4: GPA initialized with reference (n_iter > 1)
    gp_gpa_ref = GradientMaps(kernel='normalized_angle', alignment='procrustes', n_components=3, random_state=0)
    gp_gpa_ref.fit([c1, c2, c3], reference=ref_grads, n_iter=10)
    aligned_4 = gp_gpa_ref.aligned_

    # Should be different from Case 3 (Fixed Alignment)
    diff_4_3 = np.linalg.norm(aligned_4[0] - aligned_3[0])
    print(f"Diff between GPA(ref) and Fixed(ref): {diff_4_3}")
    assert diff_4_3 > 0.01, "GPA initialized with reference should differ from Fixed Alignment"

if __name__ == "__main__":
    test_alignment_methods()
