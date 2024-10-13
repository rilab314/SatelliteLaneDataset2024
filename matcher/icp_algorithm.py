import cv2
import numpy as np
import open3d as o3d

class IcpApplier:
    def __init__(self):
        self.icp_thr = 1000

    def icp_apply(self, source_image, target_image, vis=0):
        source_pcd = self.convert_pcd(source_image)
        target_pcd = self.convert_pcd(target_image)
        transform = self.icp(source_pcd, target_pcd, vis=vis)
        return transform


    def convert_pcd(self, mask):
        points = np.where(mask)
        point_zero = np.zeros_like(points[0])
        # numpy indices are (y, x) order
        points = np.stack((points[1], points[0], point_zero), axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])
        return pcd


    def icp(self, source, target, vis=0):
        # source.points [N, 3]
        translation = np.mean(target.points, axis=0) - np.mean(source.points, axis=0)
        transformation = np.eye(4)
        transformation[:3, 3] = translation
        print('init trans\n', transformation)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, self.icp_thr, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)

        evaluation = o3d.pipelines.registration.evaluate_registration(source, target, self.icp_thr, transformation)
        print(evaluation)

        source.paint_uniform_color([0, 0.651, 0.929])
        target.paint_uniform_color([1, 0.706, 0])
        if vis:
            o3d.visualization.draw_geometries([source, target])
        source.transform(reg_p2p.transformation)
        if vis:
            o3d.visualization.draw_geometries([source, target])

        transform3d = reg_p2p.transformation
        transform2d = np.zeros((2, 3))
        transform2d[:2, :2] = transform3d[:2, :2]
        transform2d[:2, 2] = transform3d[:2, 3]
        print('transform2d\n', transform2d)
        return transform2d


    def transform_image(self, image, transform):
        aligned_sema_map = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]))
        return aligned_sema_map