import pandas as pd
import numpy as np
from shapely.geometry import LineString
from io import BytesIO
from PIL import Image
import cv2
from .blob_util import plot_inliers_with_obb_html_bytes, plot_plane_inliers_outliers_html_bytes, upload_html_bytes_to_blob, upload_matplotlib_fig_to_blob, snapshot_plotly_html_bytes

def point_axis_align(df, survey_basis):
    xyz = df[['x', 'y', 'z']].values
    rotated_xyz = xyz @ survey_basis.T

    # Convert back to DataFrame
    df_rotated = pd.DataFrame(rotated_xyz, columns=['x', 'y', 'z'])
    df_concate = pd.concat([df_rotated, df.iloc[:, 3:]], axis=1)

    return df_concate

def cluster_floor_ceiling(df, eps, min_samples, type, blobs=None, min_points=10000):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    pts = df[["x", "y", "z"]].to_numpy()
    col_u8 = df[["r", "g", "b"]].to_numpy()
    col = col_u8 / 255.0

    if blobs is not None:
        snapshot_blob_client = blobs(f"{type}_cluster.html")
        snapshot_html_bytes = snapshot_plotly_html_bytes(
                points_xyz=pts,
                colors_rgb=col_u8,     # uint8 [0,255]
                point_size=2,
            )
        upload_html_bytes_to_blob(snapshot_html_bytes, snapshot_blob_client)

    feats = StandardScaler().fit_transform(np.hstack([pts, col]))  # xyz + rgb
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)

    # build cluster dict (skip noise and small clusters)
    cluster_dict = {}
    for lab in np.unique(labels):
        if lab == -1:
            continue
        m = labels == lab
        if m.sum() <= min_points:
            continue
        cluster_dict[int(lab)] = np.hstack([pts[m], col_u8[m]])  # xyz + rgb(0-255)
    return cluster_dict

def fit_ceiling_floor(
    df, 
    distance_threshold, 
    ransac_n, 
    num_iterations, 
    alpha_value, 
    logger,
    type='floor',
    blobs=None,
    snapshot_idx=None,
    ):
    import alphashape
    import matplotlib.pyplot as plt
    import open3d as o3d

    ransac_snapshot_blob_client = None
    planefit_snapshot_blob_client = None
    boundary_snapshot_blob_client = None
    edgepoints_snapshot_blob_client = None

    # If caller provided blobs + index, build snapshot clients here (unless explicitly overridden)
    if blobs is not None:
        ransac_snapshot_blob_client = blobs(f"{type}_ransac_{snapshot_idx}.html")
        planefit_snapshot_blob_client = blobs(f"{type}_planefit_{snapshot_idx}.html")
        boundary_snapshot_blob_client = blobs(f"{type}_boundary_{snapshot_idx}.png")
        edgepoints_snapshot_blob_client = blobs(f"{type}_edgepoints_{snapshot_idx}.png")

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
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Plane Fitting")
    if ransac_snapshot_blob_client is not None:
        pts = np.asarray(pcd.points)  # (N, 3)
        inliers = np.asarray(inliers, dtype=int)
        inlier_pts = pts[inliers]  # plane points
        outlier_pts = np.delete(pts, inliers, axis=0)

        snapshot_html_bytes = plot_plane_inliers_outliers_html_bytes(
            inlier_pts,
            outlier_pts,
            point_size=2,
        )
        upload_html_bytes_to_blob(snapshot_html_bytes, ransac_snapshot_blob_client)

    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
    bbox = inlier_cloud.get_oriented_bounding_box()
    bbox.color = (0, 1, 0)  # Green box
    bbox_zmin = bbox.get_min_bound()[2]  # Compute the center
    bbox_zmax = bbox.get_max_bound()[2]

    # o3d.visualization.draw_geometries([inlier_cloud, bbox])
    if planefit_snapshot_blob_client is not None:
        inlier_pts = np.asarray(inlier_cloud.points)  # (Ni, 3)

        # Oriented bounding box from inliers
        bbox = inlier_cloud.get_oriented_bounding_box()
        bbox.color = (0, 1, 0)

        html_bytes = plot_inliers_with_obb_html_bytes(inlier_pts, bbox, point_size=2)
        upload_html_bytes_to_blob(html_bytes, planefit_snapshot_blob_client)

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
        logger.info("Alpha shape is not a Polygon or MultiPolygon.")
        boundary_coords = []

    # Plot if boundary was found
    if boundary_coords:
        boundary_array = np.array(boundary_coords)
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], s=10, label="Floor Points")
        ax.plot(boundary_array[:, 0], boundary_array[:, 1], "r-", linewidth=2, label="Boundary")
        ax.scatter(boundary_array[:, 0], boundary_array[:, 1], s=1, color="red", label="Corner Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Alpha Shape Boundary (alpha={alpha_value})")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        if boundary_snapshot_blob_client is not None:
            upload_matplotlib_fig_to_blob(fig, boundary_snapshot_blob_client)
    else:
        logger.info("No valid boundary found.")
    
    line = LineString(boundary_coords)
    simplified = line.simplify(tolerance=0.8)  # tweak tolerance

    corner_coords = np.array(simplified.coords)

    extruded_coords = []
    for x, y in corner_coords:
        extruded_coords.append([x, y, bbox_zmin])
        extruded_coords.append([x, y, bbox_zmax])

    # Plot
    fig, ax = plt.subplots()
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.scatter(corner_coords[:, 0], corner_coords[:, 1], color="red", label="Corner Points")
    plt.title('Alpha Shape (Concave Hull)')

    if edgepoints_snapshot_blob_client is not None:
        upload_matplotlib_fig_to_blob(fig, edgepoints_snapshot_blob_client)

    return bbox_zmin, bbox_zmax, extruded_coords

def sorted_merged_floor_ceiling_plane(bbox_arr):
    plane_arr = sorted(bbox_arr, key=lambda x: x[1]) #sort by zmin
    return plane_arr

def points_between_level(bbox1, bbox2, xyzrgb):
    z_min_floor = bbox1[1]
    z_max_floor = bbox2[0]
    # Filter points between the floors
    filtered_points = xyzrgb[(xyzrgb[:, 2] > z_min_floor) & (xyzrgb[:, 2] < z_max_floor)]
    return filtered_points

def project_points_to_floor(filtered_points, bins):
    import matplotlib.pyplot as plt

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
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    img_bytes.seek(0) 
    img = Image.open(img_bytes)
    img_array = np.array(img)
    bgr_arr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    return xy_projected, x_edges, y_edges, bgr_arr

