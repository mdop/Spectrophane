

class StackGenerator:
    """Generates stack combinations based on a general configuration in the constructor. Individual functions output stack combinations based on the function rules."""
    def __init__(self, num_materials: int):
        pass

    def full_homogeneous_layered_stacks(layer_count: int, layer_thickness: float, stack_order: list | None = None):
        """generates StackData for all possible combinations of layer numbers of materials without regard of layer order. 
        Will stack materials in order provided in stack_order or by default ascending. 
        This generator is targeted towards transmission stacks that are not or less dependent on layer order."""
        #check stack_order if provided
        pass