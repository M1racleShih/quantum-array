from qarray import QArray

if __name__ == "__main__":
    # Create 4x6 Rectangular Array (index_base=1)
    A = QArray(4, 6, index_base=1, topology="rect")
    C_rect = A.couplers
    
    # Test Data
    # q8 is at (1, 1). 
    # Right neighbor is q9  -> c8-9  (Stored at 1,1, Channel 0)
    # Down neighbor is q14 -> c8-14 (Stored at 1,1, Channel 1)
    
    print("=== Test get_loc ===")
    
    lbl1 = "c8-9"
    loc1 = C_rect.get_loc(lbl1)
    print(f"Location of {lbl1} : {loc1}")
    # Expected: (1, 1, 0) -> Row 1, Col 1, Horizontal Channel
    
    lbl2 = "c8-14"
    loc2 = C_rect.get_loc(lbl2)
    print(f"Location of {lbl2}: {loc2}")
    # Expected: (1, 1, 1) -> Row 1, Col 1, Vertical Channel
    
    lbl3 = "c99-100"
    loc3 = C_rect.get_loc(lbl3)
    print(f"Location of {lbl3}: {loc3}")
    # Expected: None
    
    # Verify consistency
    if loc1:
        r, c, k = loc1
        print(f"Verify C[{r}, {c}, {k}] == {lbl1}: {C_rect[r,c,k] == lbl1}")
    
    # 4x6 Rectangular Array
    # q8 is at (1, 1) assuming base=1
    # Neighbors: q2(Up), q14(Down), q7(Left), q9(Right)
    A = QArray(4, 6, index_base=1, topology="rect")
    
    C_rect = A.couplers
    
    print("=== Matrix Shape ===")
    print(C_rect._data.shape) # (4, 6, 2)

    print("\n=== Couplers of q8 (Internal Node) ===")
    # q8 index=7 -> (1, 1)
    # Should find 4 couplers
    print(C_rect.couplers_of("q8"))
    # Expected:
    # Right (Source): c8-9  (stored at [1, 1, 0])
    # Down  (Source): c8-14 (stored at [1, 1, 1])
    # Left  (Target): c7-8  (stored at [1, 0, 0])
    # Up    (Target): c2-8  (stored at [0, 1, 1])
    
    print("\n=== Couplers of q1 (Corner Node) ===")
    # q1 index=0 -> (0, 0)
    # Should find 2 couplers (Right, Down)
    print(C_rect.couplers_of("q1"))
    
    print("\n=== Slice Access ===")
    # Get all horizontal couplers in the first row
    # row=0, all cols, channel=0
    print(C_rect[0, :, 0])
