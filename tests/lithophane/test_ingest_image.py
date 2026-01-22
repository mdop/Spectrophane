import pytest
import numpy as np
from PIL import Image
from spectrophane.lithophane.ingest_image import format_image, image_to_stackmap, stackmap_to_voxelmap, _pixel_voxel_stack_height_matching
from spectrophane.core.dataclasses import StackCandidates, StackData, VoxelGeometry
from spectrophane.color.conversions import linrgb_to_xyz, decode_rgb

class MockInverter:
    def invert_color(self, colors: np.ndarray, max_suggested_stacks: int, color_space: str):
        mat_num = np.random.randint(0,5, size=(len(colors), 10))
        thicknesses = np.array([[0.2, 0.1, 0.3, 0.5, 0.0, 0.2, 0.0, 0.3, 0.5, 1.0]]*len(colors))
        return StackCandidates(material_nums=mat_num, thicknesses=thicknesses, rgb=colors), np.indices([len(colors)]), np.random.random(len(colors))

@pytest.fixture
def mock_image():
    img = np.random.randint(0,255,size=(90,80,3)) #(h*b*channel)
    img[1,1,:] = img[0,0,:]
    return img

def test_format_image():
    # Create a sample image
    image = Image.new("RGB", (100, 100), (255, 0, 0))
    resolution = (50, 50)
    
    # Test format_image
    formatted_image = format_image(image, resolution)
    assert formatted_image.shape == (50, 50, 3)

def test_image_to_stackmap(mock_image):
    # Test image_to_stackmap
    unique_stacks, image_unique_color_indexes, calc_score = image_to_stackmap(mock_image, MockInverter(), convert_xyz=True)
    
    assert isinstance(unique_stacks, StackCandidates)
    assert len(unique_stacks.material_nums) < mock_image.shape[0]*mock_image.shape[1] #a duplicate was deliberately introduced in fixture
    assert isinstance(image_unique_color_indexes, np.ndarray)
    assert image_unique_color_indexes.shape == mock_image.shape[0:2]
    assert isinstance(calc_score, np.ndarray)
    assert calc_score.shape == mock_image.shape[0:2]

def test_pixel_voxel_stack_height_matching():
    #basic case
    cumulative_voxel_heights = np.array([0.1, 0.2, 0.3])
    cumulative_stack_heights = np.array([0.1, 0.3])
    stack_material_nums = np.array([0, 1])
    
    result = _pixel_voxel_stack_height_matching(cumulative_voxel_heights, cumulative_stack_heights, stack_material_nums)
    assert np.all(result == np.array([0, 1, 1]))
    
    #multiple stack matching
    cumulative_voxel_heights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cumulative_stack_heights = np.array([0.1, 0.2, 0.5])
    stack_material_nums = np.array([5, 1, 2])
    
    result = _pixel_voxel_stack_height_matching(cumulative_voxel_heights, cumulative_stack_heights, stack_material_nums)
    assert np.array_equal(result, np.array([5, 1, 2, 2, 2]))

def test_pixel_voxel_stack_height_matching_empty_layer():
    cumulative_voxel_heights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    cumulative_stack_heights = np.array([0.1, 0.2, 0.2, 0.5])
    stack_material_nums = np.array([5, 1, 0, 2])
    
    result = _pixel_voxel_stack_height_matching(cumulative_voxel_heights, cumulative_stack_heights, stack_material_nums)
    assert np.array_equal(result, np.array([5, 1, 2, 2, 2])) #should not contain 0 as the layer has 0 thickness


def test_stackmap_to_voxelmap():
    # Create sample inputs
    layer_thicknesses = np.array([0.1, 0.1, 0.2, 0.1])
    voxel_size_xy = (10, 10)
    stacks = StackData(thicknesses=np.array([[0.2, 0.3], [0.1, 0.4], [0.5, 0.0]]), material_nums=np.array([[0, 1], [1, 2], [1, 0]]))
    image_stack_indexes = np.random.randint(0,3, voxel_size_xy)
    
    # Test stackmap_to_voxelmap
    voxel_geometry = stackmap_to_voxelmap(layer_thicknesses, voxel_size_xy, stacks, image_stack_indexes)
    
    assert isinstance(voxel_geometry, VoxelGeometry), "Result should be an instance of VoxelGeometry"
    assert voxel_geometry.materials.shape == (10, 10, 4), "material_map should have the correct shape"
    assert not np.any(voxel_geometry.materials == -1), "No unassigned voxel positions should be present"

