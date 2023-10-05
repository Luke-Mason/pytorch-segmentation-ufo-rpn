
class SlidingWindowConfig:
    def __init__(self, name: str, kernel_3d: Tuple[int, int, int], stride_3d: Tuple[int, int, int]):
        if name == 'max':
            self.strategy_fn = lambda x: np.max(x)
        elif name == 'mean':
            self.strategy_fn = lambda x: np.mean(x)
        elif name == 'min':
            self.strategy_fn = lambda x: np.min(x)
        elif name == 'sum':
            self.strategy_fn = lambda x: np.sum(x)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

        self.name = name
        self.kernel_3d = kernel_3d
        self.stride_3d = stride_3d

def sliding_window_3d(arr, config: SlidingWindowConfig):
    kernel_shape = config.kernel_3d
    stride = config.stride_3d
    func = config.strategy_fn

    # Get array shape and kernel shape
    arr_shape = arr.shape
    kernel_shape = np.array(kernel_shape)
    stride = np.array(stride)

    # Calculate output shape
    output_shape = ((arr_shape - kernel_shape) // stride) + 1

    # Initialize an array to store the results
    results = np.zeros(output_shape)

    # Iterate over the array with the specified stride
    for i in range(0, arr_shape[0] - kernel_shape[0] + 1, stride[0]):
        for j in range(0, arr_shape[1] - kernel_shape[1] + 1, stride[1]):
            for k in range(0, arr_shape[2] - kernel_shape[2] + 1, stride[2]):
                # Extract the subarray within the sliding window
                subarray = arr[i:i+kernel_shape[0], j:j+kernel_shape[1], k:k+kernel_shape[2]]

                # Apply the lambda function to the subarray and store the result
                results[i//stride[0], j//stride[1], k//stride[2]] = func(subarray)

    return results


def extract_mask_values_using_polygons(mask: np.ndarray,
                                       polygons: MultiPolygon):
    """ Return numpy mask for given polygons.
        polygons should already be converted to image coordinates.
        non values are given -1.
    """
    # Mark the values to extract with a 1.
    mark_value = 1
    marked_mask = np.zeros(mask.shape, dtype=np.int8)
    cv2.fillPoly(marked_mask, polygons, mark_value)

    # Extract the values from the main mask using the marked mask
    extracted_values_mask = np.full(marked_mask.shape, -1, dtype=np.float32)
    for index, element in np.ndenumerate(marked_mask):
        if element == mark_value:
            extracted_values_mask[index] = mask[index]
    return extracted_values_mask

def mask_for_polygons(
        im_size: Tuple[int, int], polygons: MultiPolygon) -> np.ndarray:
    """ Return numpy mask for given polygons.
    polygons should already be converted to image coordinates.
    """
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
    interiors = [int_coords(pi.coords) for poly in polygons.geoms
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask