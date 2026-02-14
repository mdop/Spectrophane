import pytest
import numpy as np
from spectrophane.core.dataclasses import VoxelGeometry, StackCandidates
from spectrophane.lithophane.pipeline import export_geometry, generate_lithophane_from_image
from spectrophane.lithophane.ingest_image import image_to_stackmap, stackmap_to_voxelmap
from spectrophane.lithophane.solid_generation import PerVoxelBoxBuilder
from spectrophane.lithophane.export import STLTessellationBackend
from spectrophane.inverse.inverter import Inverter
from PIL import Image, ImageDraw

@pytest.fixture
def mock_geometry():
    # Mock VoxelGeometry with sample data
    return VoxelGeometry(
        materials=np.array([[[0,1],[0,0]],[[1,1],[1,0]]]),
        layer_thickness=[0.1,0.2],
        voxel_size_xy=[0.5,0.5],
        material_names=["MaterialA", "MaterialB"]
    )

class MockInverter:
    preferred_color_space = "xyz"
    def invert_color(self, colors: np.ndarray, max_suggested_stacks: int = 1, color_space: str = "xyz") -> tuple[StackCandidates, np.ndarray, np.ndarray]:
        stacks = np.array([[0,0],[0,1]])
        rgbs = np.array([[1,1,1],[254,254,254]])
        assignment = np.astype(np.round(np.average(colors, axis=1), decimals=0), np.uint8)
        return (StackCandidates(stacks[assignment], np.array([[0.1,0.2]]*len(assignment)), rgbs[assignment]), 
                np.indices(assignment),
                np.random.random_sample(len(assignment)))


@pytest.fixture
def mock_lut_inverter():
    return MockInverter()

@pytest.fixture
def mock_inverter():
    return MockInverter()

@pytest.fixture
def mock_image():
    # Create a simple black and white image
    image = Image.new('RGB', (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 20, 20), fill=(0, 0, 0))
    return image

def test_export_geometry(mock_geometry):
    # Mock SolidBuilder and SolidBackend
    class MockSolidBuilder:
        def solids_for_material(self, geometry, material_id):
            return [f"Solid_{material_id}"]

    class MockExportBackend:
        def begin(self, material_id):
            pass

        def add(self, solid):
            pass

        def end(self):
            return [f"output_{material_id}.stl" for material_id in [0, 1]]

    builder = MockSolidBuilder()
    export_backend = MockExportBackend()

    output_paths = export_geometry(mock_geometry, builder, export_backend)
    assert len(output_paths) == 2
    assert "output_0.stl" in output_paths
    assert "output_1.stl" in output_paths

def test_transform_image_integrationtest(tmp_path, mock_image, mock_inverter):
    output_basepath = tmp_path / "test.stl" #check if optional file extension is correctly handled
    material_names = ["MaterialA", "MaterialB"]
    target_width = 30
    target_height = 20
    output_files, expected_rgb_img, score_img = generate_lithophane_from_image(mock_image, (target_width, target_height),
                                                                mock_inverter,
                                                                np.array([0.1,0.2]), (0.4,0.4),
                                                                material_names,
                                                                PerVoxelBoxBuilder(),
                                                                STLTessellationBackend(output_basepath, material_names, binary=True))
    
    assert len(output_files) == 2
    assert output_files[0].endswith("test_MaterialA.stl")
    assert output_files[1].endswith("test_MaterialB.stl")
    assert expected_rgb_img.shape == (target_height,target_width,3)
    assert score_img.shape == (target_height,target_width)

    