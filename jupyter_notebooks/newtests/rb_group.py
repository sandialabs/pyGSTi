from pygsti.extras import rb

def test_rb_group():
    
    # Tests the key aspects of the group module by creating
    # the 1Q clifford group
    rb.group.construct_1Q_Clifford_group()
    
    return