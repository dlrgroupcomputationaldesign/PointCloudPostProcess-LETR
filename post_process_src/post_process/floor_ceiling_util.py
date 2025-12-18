import pandas as pd
import open3d as o3d
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from io import BytesIO
from PIL import Image
import cv2

def point_axis_align(df, survey_basis):
    xyz = df[['x', 'y', 'z']].values
    rotated_xyz = xyz @ survey_basis.T

    # Convert back to DataFrame
    df_rotated = pd.DataFrame(rotated_xyz, columns=['x', 'y', 'z'])
    df_concate = pd.concat([df_rotated, df.iloc[:, 3:]], axis=1)

    return df_concate

def cluster_floor_ceiling(df, eps, min_samples):
    # Extract xyzrgb columns (x, y, z, r, g, b)
    points = df[['x', 'y', 'z']].values
    colors = df[['r', 'g', 'b']].values
    colors_nor = df[['r', 'g', 'b']].values / 255.0  # Normalize RGB values to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_nor)  # Assign RGB colors
    o3d.visualization.draw_geometries([pcd])                                        # Visualize point cloud

    # Optionally: Normalize the features for DBSCAN (x, y, z, r, g, b)
    features = np.hstack([points, colors_nor])  # Use x, y, z, r, g, b as features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust parameters as needed
    cluster_num = dbscan.fit_predict(features_scaled)

    # Count the number of clusters (excluding noise points, labeled as -1)
    unique_num, counts = np.unique(cluster_num, return_counts=True)
    
    # Exclude noise (-1) & number of points < 10000 from cluster count
    cluster_dict = {label: np.hstack([points[cluster_num == label], colors[cluster_num == label]])for idx, label in enumerate(unique_num) if label != -1 and counts[idx] > 10000}
            
    return cluster_dict

def fit_ceiling_floor(df, distance_threshold, ransac_n, num_iterations, alpha_value):
    # Extract XYZ and RGB columns
    xyz = df[['x', 'y', 'z']].values  # Point coordinates
    rgb = df[['r', 'g', 'b']].values / 255.0  # Normalize RGB values (0-1)
    points = df[['x', 'y']].values
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)  # Assign RGB colors

    # Apply RANSAC plane fitting
    plane_model, inliers = pcd.segment_plane(distance_threshold,  # Adjust threshold as needed
                                            ransac_n,
                                            num_iterations)

    # Extract plane parameters (ax + by + cz + d = 0)
    a, b, c, d = plane_model
    # print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    # Extract inlier and outlier points
    inlier_cloud = pcd.select_by_index(inliers)  # Points belonging to the plane
    outlier_cloud = pcd.select_by_index(inliers, invert=True)  # Points not in the plane

    # Color the plane points (for visualization)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red for plane
    outlier_cloud.paint_uniform_color([0, 0, 1.0])  # Blue for non-plane

    # Visualize the result
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Plane Fitting")

    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
    bbox = pcd.get_oriented_bounding_box()
    bbox.color = (0, 1, 0)  # Green box
    bbox_zmin = bbox.get_min_bound()[2]  # Compute the center
    bbox_zmax = bbox.get_max_bound()[2]
    o3d.visualization.draw_geometries([inlier_cloud, bbox])

    floor_points = np.asarray(pcd.points)[inliers]
    floor_points_2d = floor_points[:, :2]

    # Choose an appropriate alpha value. This is crucial and might require experimentation.
    # A smaller alpha will result in a tighter, more detailed (potentially fragmented) shape.
    # A larger alpha will approach the convex hull.
    
    # Create an alpha shape
    alpha_shape = alphashape.alphashape(floor_points_2d, alpha_value)
    if alpha_shape.geom_type == 'Polygon':
        boundary_coords = alpha_shape.exterior.coords
    elif alpha_shape.geom_type == 'MultiPolygon':
        boundary_coords = max(alpha_shape.geoms, key=lambda p: p.area).exterior.coords
    else:
        print("Alpha shape is not a Polygon or MultiPolygon.")
        boundary_coords = []

    # Plot if boundary was found
    if boundary_coords:
        boundary_array = np.array(boundary_coords)
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], s=10, label='Floor Points')
        plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'r-', linewidth=2, label='Boundary')
        plt.scatter(boundary_array[:, 0], boundary_array[:, 1], s=1, color='red', label='Corner Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Alpha Shape Boundary (alpha={alpha_value})')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    else:
        print("No valid boundary found.")

    
    line = LineString(boundary_coords)
    simplified = line.simplify(tolerance=0.8)  # tweak tolerance

    corner_coords = np.array(simplified.coords)

    extruded_coords = []
    for x, y in corner_coords:
        extruded_coords.append([x, y, bbox_zmin])
        extruded_coords.append([x, y, bbox_zmax])

    # Plot
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.scatter(corner_coords[:, 0], corner_coords[:, 1], color="red", label="Corner Points")
    plt.title('Alpha Shape (Concave Hull)')
    plt.show()

    return bbox_zmin, bbox_zmax, extruded_coords

def sorted_merged_floor_ceiling_plane(bbox_arr):
    sorted_data = sorted(bbox_arr, key=lambda x: x[1]) #sort by zmin
    print('sorted', sorted_data)
    plane_arr = sorted_data
    # Create the result array
    # plane_arr = []
    # i = 0
    # while i < len(sorted_data):
    #     current = sorted_data[i]
    #     plane_arr.append([current[1], current[2]])
    #     i += 1
    # print('merged', plane_arr)
    return plane_arr

def points_between_level(bbox1, bbox2, xyzrgb):
    z_min_floor = bbox1[1]
    z_max_floor = bbox2[0]
    # Filter points between the floors
    filtered_points = xyzrgb[(xyzrgb[:, 2] > z_min_floor) & (xyzrgb[:, 2] < z_max_floor)]
    return filtered_points

def project_points_to_floor(filtered_points, bins):
    # Project to XY plane (ignore Z)
    xy_projected = filtered_points[:, :2]  # Keep only x, y
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(xy_projected[:, 0], xy_projected[:, 1], bins=bins)

    # Plot histogram
    plt.figure(figsize=(10, 8))
    plt.imshow(hist.T, origin='lower', cmap='hot', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.axis('off')

    # Save plot to a BytesIO object
    img_bytes = BytesIO()
    # Save figure without extra white space
    # img_path = 'bridger.png'
    # plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    img_bytes.seek(0) 
    img = Image.open(img_bytes)
    img_array = np.array(img)
    bgr_arr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    return xy_projected, x_edges, y_edges, bgr_arr

