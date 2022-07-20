from .ransac import ransac
from .utils import dist_matrix, orientation_diff
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.collections
from scipy.spatial import Delaunay


def select_seeds(dist1: torch.Tensor, R1: float, dist2: torch.Tensor, R2: float,scores1: torch.Tensor, scores21: torch.Tensor,distance:torch.Tensor, fnn12: torch.Tensor, fnn21: torch.Tensor, mnn: torch.Tensor):
    """
        Select seed correspondences among the set of available matches.

        dist1: Precomputed distance matrix between keypoints in image I_1
        R1: Base radius of neighborhoods in image I_1
        scores1: Confidence scores on the putative_matches. Usually holds Lowe's ratio scores.
        fnn12: Matches between keypoints of I_1 and I_2.
               The i-th entry of fnn12 is j if and only if keypoint k_i in image I_1 is matched to keypoint k_j in image I_2
        mnn: A mask indicating which putative matches are also mutual nearest neighbors. See documentation on 'force_seed_mnn' in the DEFAULT_CONFIG.
             If None, it disables the mutual nearest neighbor filtering on seed point selection.
             Expected a bool tensor with shape (num_keypoints_in_source_image,)

        Returns:
            Indices of seed points.

            im1seeds: Keypoint index of chosen seeds in image I_1
            im2seeds: Keypoint index of chosen seeds in image I_2
    """
    im1neighmap = dist1 < R1 ** 2  # (n1, n1)
    im2neighmap = dist2 < R2 ** 2  # Use true\false to indicate whether it is a neighborhood
    # find out who scores higher than whom
    f_im1 = torch.cumsum(im1neighmap.type(torch.int), dim=1)
    f_im2 = torch.cumsum(im2neighmap.type(torch.int), dim=1)
    fnum_im1, indf = torch.max(f_im1, dim=1)  # The number of features in the neighborhood in Figure 1
    fnum_im2, indf2 = torch.max(f_im2, dim=1)  # The number of features in the neighborhood in Figure 2
    common = (im1neighmap & im2neighmap).type(torch.int)
    count = torch.cumsum(common, dim=1)
    neigh_counts, ind = torch.max(count, dim=1)
    neigh_simi = 1 - neigh_counts / ((fnum_im1 + fnum_im2) * 0.5) #Neighborhood similarity

    #Adaptive Neighborhood Radius
    simi = (fnum_im1 + fnum_im2)/(np.pi*(R1**2+R2**2))

    #two-way ratio test
    for i in range(len(fnn12)):
        if fnn21[fnn12[i]] == i:
            scores1[i] = math.sqrt(scores1[i] * scores21[fnn12[i]])
        else:
            scores1[i] = math.sqrt(scores1[i])

    im1scorescomp = scores1.unsqueeze(1) > scores1.unsqueeze(0)  # (n1, n1)
    im1scorescomp2 = neigh_simi.unsqueeze(1) > neigh_simi.unsqueeze(0)



    # find out who scores higher than all of its neighbors: seed points
    if mnn is not None:
        im1bs = (~torch.any(im1neighmap  & im1scorescomp & im1scorescomp2  & mnn.unsqueeze(0),
                            dim=1)) & mnn & (scores1 < 0.8**2)  # (n1,)
    else:
        im1bs = (~torch.any(im1neighmap & im1scorescomp & im1scorescomp2, dim=1)) & (scores1 <
                                                                    0.8**2)

    # collect all seeds in both images and the 1NN of the seeds of the other image
    im1seeds = torch.where(im1bs)[0]  # (n1bs) index format
    im2seeds = fnn12[im1bs]  # (n1bs) index format

    return im1seeds, im2seeds,scores1,simi


def extract_neighborhood_sets(
        o1: torch.Tensor, o2: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
        dist1: torch.Tensor, im1seeds: torch.Tensor, im2seeds: torch.Tensor,
        k1: torch.Tensor, k2: torch.Tensor, R1: torch.Tensor, R2: torch.Tensor,
        fnn12: torch.Tensor, ORIENTATION_THR: float, SCALE_RATE_THR: float,
        SEARCH_EXP: float, MIN_INLIERS: float):
    """
        Assign keypoints to seed points. This checks both the distance and
        the agreement of the local transformation if available.

        o1: Orientations of keypoints in image I_1
        o2: Orientations of keypoints in image I_2
        s1: Scales of keypoints in image I_1
        s2: Scales of keypoints in image I_2
        dist1: Precomputed distance matrix between keypoints in image I_1
        im1seeds: Keypoint index of chosen seeds in image I_1
        im2seeds: Keypoint index of chosen seeds in image I_2
        k1: Keypoint locations in image I_1
        k2: Keypoint locations in image I_2
        R1: Base radius of neighborhoods in image I_1
        R2: Base radius of neighborhoods in image I_2
        fnn12: Matches between keypoints of I_1 and I_2.
               The i-th entry of fnn12 is j if and only if keypoint k_i in image I_1 is matched to keypoint k_j in image I_2
        ORIENTATION_THR: Maximum deviation of orientation with respect to seed S_i to keep a keypoint in i-th neighborhood
        SCALE_RATE_THR: Maximum deviation of scale with respect to seed S_i to keep a keypoint in i-th neighborhood
        SEARCH_EXP: Expansion rate for both radii R1 and R2 to consider inclusion of neighboring keypoints
        MIN_INLIERS: Minimum number of inliers to keep a seed point. This is used as an early filter here
                     to remove already seeds with not enough samples to ever pass this threshold.

        Returns:
            Local neighborhoods assignments:

            local_neighs_mask: Boolean matrix of size (num_seeds, num_keypoints).
                               Entry (i, j) is True iff keypoint j was assigned to seed i.
            rdims: Number of keypoints included in the neighborhood for each seed
            im1seeds: Keypoint index of chosen seeds in image I_1
            im2seeds: Keypoint index of chosen seeds in image I_2

    """

    dst1 = dist1[im1seeds, :]
    dst2 = dist_matrix(k2[fnn12[im1seeds]], k2[fnn12])

    # initial candidates are matches which are close to the same seed in both images
    ther1 = torch.Tensor([(SEARCH_EXP * r1) **2 for r1 in R1]).reshape(len(R1),-1).cuda(0)
    ther2 = torch.Tensor([(SEARCH_EXP * r2) ** 2 for r2 in R2]).reshape(len(R2),-1).cuda(0)
    local_neighs_mask = (dst1 < ther1) & (dst2 < ther2)


    # If requested, also their orientation delta should be compatible with that of the corresponding seed
    if ORIENTATION_THR is not None and ORIENTATION_THR < 180:
        relo = orientation_diff(o1, o2[fnn12])
        orientation_diffs = torch.abs(
            orientation_diff(relo.unsqueeze(0), relo[im1seeds].unsqueeze(1)))
        local_neighs_mask = local_neighs_mask & (orientation_diffs <
                                                 ORIENTATION_THR)

    # If requested, also their scale delta should be compatible with that of the corresponding seed
    if SCALE_RATE_THR is not None and SCALE_RATE_THR < 10:
        rels = s2[fnn12] / s1
        scale_rates = rels[im1seeds].unsqueeze(1) / rels.unsqueeze(0)
        local_neighs_mask = local_neighs_mask & (scale_rates < SCALE_RATE_THR) \
                            & (scale_rates > 1 / SCALE_RATE_THR)  # (ns, n1)

    # count how many keypoints ended up in each neighborhood
    numn1 = torch.sum(local_neighs_mask, dim=1)
    # and only keep the ones that have enough points
    valid_seeds = numn1 >= MIN_INLIERS

    local_neighs_mask = local_neighs_mask[valid_seeds, :]

    rdims = numn1[valid_seeds]

    return local_neighs_mask, rdims, im1seeds[valid_seeds], im2seeds[
        valid_seeds], R1[valid_seeds], R2[valid_seeds]


def extract_local_patterns(
        fnn12: torch.Tensor,
        fnn_to_seed_local_consistency_map_corr: torch.Tensor, k1: torch.Tensor,
        k2: torch.Tensor, im1seeds: torch.Tensor, im2seeds: torch.Tensor,
        scores: torch.Tensor,R1:torch.Tensor,R2:torch.Tensor,SEARCH_EXP: float):
    """
        Prepare local neighborhoods around each seed for the parallel RANSACs. This involves two steps:
            1) Collect all selected keypoints and refer them with respect to their seed point
            2) Sort keypoints by score for the progressive sampling to pick the best samples first

        fnn12: Matches between keypoints of I_1 and I_2.
               The i-th entry of fnn12 is j if and only if keypoint k_i in image I_1 is matched to keypoint k_j in image I_2
        fnn_to_seed_local_consistency_map_corr: Boolean matrix of size (num_seeds, num_keypoints).
                                                Entry (i, j) is True iff keypoint j was assigned to seed i.
        k1: Keypoint locations in image I_1
        k2: Keypoint locations in image I_2
        im1seeds: Keypoint index of chosen seeds in image I_1
        im2seeds: Keypoint index of chosen seeds in image I_2
        scores: Scores to rank correspondences by confidence.
                Lower scores are assumed to be more confident, consistently with Lowe's ratio scores.
                Note: scores should be between 0 and 1 for this function to work as expected.

        Returns:
            All information required for running the parallel RANSACs.
            Data is formatted so that all inputs for different RANSACs are concatenated
                along the same dimension to support different input sizes.

            im1loc: Keypoint locations in image I_1 for each RANSAC sample.
            im2loc: Keypoint locations in image I_2 for each RANSAC sample.
            ransidx: Integer identifier of the RANSAC problem.
                     This allows to distinguish inputs belonging to the same problem.
            tokp1: Index of the original keypoint in image I_1 for each RANSAC sample.
            tokp2: Index of the original keypoint in image I_2 for each RANSAC sample.
    """
    # first get an indexing representation of the assignments:
    # - ransidx holds the index of the seed for each assignment
    # - tokp1 holds the index of the keypoint in image I_1 for each assignment 
    ransidx, tokp1 = torch.where(fnn_to_seed_local_consistency_map_corr)
    # - and of course tokp2 holds the index of the corresponding keypoint in image I_2
    tokp2 = fnn12[tokp1]

    # Now take the locations in the image of each considered keypoint ... 
    im1abspattern = k1[tokp1]
    im2abspattern = k2[tokp2]
    R1 = torch.Tensor([r1 * SEARCH_EXP for r1 in R1]).cuda(0)
    R2 = torch.Tensor([r2 * SEARCH_EXP for r2 in R2]).cuda(0)

    # ... and subtract the location of its corresponding seed to get relative coordinates
    im1loc = (im1abspattern - k1[im1seeds[ransidx]])/R1[ransidx].reshape(-1,1)
    im2loc = (im2abspattern - k2[im2seeds[ransidx]])/R2[ransidx].reshape(-1,1)#


    # Finally we need to sort keypoints by scores in a way that assignments to the same seed are close together
    # To achieve this we assume scores lie in (0, 1) and add the integer index of the corresponding seed
    expanded_local_scores = scores[tokp1] + ransidx.type(scores.dtype)

    sorting_perm = torch.argsort(expanded_local_scores)

    im1loc = im1loc[sorting_perm]
    im2loc = im2loc[sorting_perm]
    tokp1 = tokp1[sorting_perm]
    tokp2 = tokp2[sorting_perm]

    return im1loc, im2loc, ransidx, tokp1, tokp2

# Judgment line intersection
def isCross(a,b,c,d):
    if abs(b[1]-a[1])/(abs(b[0]-a[0])+1e-10)==abs(d[1]-c[1])/(abs(d[0]-c[0])+1e-10):
        return False
    if max(c[0],d[0])<min(a[0],b[0]) or max(a[0],b[0])<min(c[0],d[0]) or max(c[1],d[1])<min(a[1],b[1]) or max(a[1],b[1])<min(c[1],d[1]):
        return False
    if np.dot((a - d),(c - d))*np.dot((b - d),(c - d)) > 0 or np.dot((c - b),(a - b))*np.dot((d - b),(a - b)) > 0:
        return False
    if (a[0]==c[0] and a[1]==c[1]) or (a[0]==d[0] and a[1]==d[1]) or (b[0]==c[0] and b[1]==c[1]) or (b[0]==d[0] and b[1]==d[1]):
        return False
    return True

# Find the number of intersections
def crossWithLineset(line,lineset,seeds2):
    sum = 0
    for t in lineset:
        flag = isCross(seeds2[t[0]], seeds2[t[1]], seeds2[line[0]], seeds2[line[1]])
        if flag == True:
            sum += 1
    return sum



def adalam_core(k1: torch.Tensor,
                k2: torch.Tensor,
                fnn12: torch.Tensor,
                fnn21: torch.Tensor,
                scores1: torch.Tensor,
                scores21: torch.Tensor,
                distance: torch.Tensor,
                config: dict,
                mnn: torch.Tensor = None,
                im1shape: tuple = None,
                im2shape: tuple = None,
                o1: torch.Tensor = None,
                o2: torch.Tensor = None,
                s1: torch.Tensor = None,
                s2: torch.Tensor = None,):
    """
        Call the core functionality of AdaLAM, i.e. just outlier filtering. No sanity check is performed on the inputs.

        Inputs:
            k1: keypoint locations in the source image, in pixel coordinates.
                Expected a float32 tensor with shape (num_keypoints_in_source_image, 2).
            k2: keypoint locations in the destination image, in pixel coordinates.
                Expected a float32 tensor with shape (num_keypoints_in_destination_image, 2).
            fn12: Initial set of putative matches to be filtered.
                  The current implementation assumes that these are unfiltered nearest neighbor matches,
                  so it requires this to be a list of indices a_i such that the source keypoint i is associated to the destination keypoint a_i.
                  For now to use AdaLAM on different inputs a workaround on the input format is required.
                  Expected a long tensor with shape (num_keypoints_in_source_image,).
            scores1: Confidence scores on the putative_matches. Usually holds Lowe's ratio scores.
            mnn: A mask indicating which putative matches are also mutual nearest neighbors. See documentation on 'force_seed_mnn' in the DEFAULT_CONFIG.
                 If None, it disables the mutual nearest neighbor filtering on seed point selection.
                 Expected a bool tensor with shape (num_keypoints_in_source_image,)
            im1shape: Shape of the source image. If None, it is inferred from keypoints max and min, at the cost of wasted runtime. So please provide it.
                      Expected a tuple with (width, height) or (height, width) of source image
            im2shape: Shape of the destination image. If None, it is inferred from keypoints max and min, at the cost of wasted runtime. So please provide it.
                      Expected a tuple with (width, height) or (height, width) of destination image
            o1/o2: keypoint orientations in degrees. They can be None if 'orientation_difference_threshold' in config is set to None.
                   See documentation on 'orientation_difference_threshold' in the DEFAULT_CONFIG.
                   Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)
            s1/s2: keypoint scales. They can be None if 'scale_rate_threshold' in config is set to None.
                   See documentation on 'scale_rate_threshold' in the DEFAULT_CONFIG.
                   Expected a float32 tensor with shape (num_keypoints_in_source/destination_image,)

        Returns:
            Filtered putative matches.
            A long tensor with shape (num_filtered_matches, 2) with indices of corresponding keypoints in k1 and k2.
    """
    AREA_RATIO = config['area_ratio']
    SEARCH_EXP = config['search_expansion']
    RANSAC_ITERS = config['ransac_iters']
    MIN_INLIERS = config['min_inliers']
    MIN_CONF = config['min_confidence']
    ORIENTATION_THR = config['orientation_difference_threshold']
    SCALE_RATE_THR = config['scale_rate_threshold']
    REFIT = config['refit']

    if im1shape is None:
        k1mins, _ = torch.min(k1, dim=0)
        k1maxs, _ = torch.max(k1, dim=0)
        im1shape = (k1maxs - k1mins).cpu().numpy()
    if im2shape is None:
        k2mins, _ = torch.min(k2, dim=0)
        k2maxs, _ = torch.max(k2, dim=0)
        im2shape = (k2maxs - k2mins).cpu().numpy()

    # Compute seed selection radii to be invariant to image rescaling
    R1 = np.sqrt(np.prod(im1shape[:2]) / AREA_RATIO / np.pi)
    R2 = np.sqrt(np.prod(im2shape[:2]) / AREA_RATIO / np.pi)

    # Precompute the inner distances of keypoints in image I_1
    dist1 = dist_matrix(k1, k1)
    corr = k2[fnn12]
    dist2 = dist_matrix(corr, corr)

    # Select seeds
    im1seeds, im2seeds,scores,neigh_simi = select_seeds(dist1, R1,dist2, R2, scores1, scores21,distance, fnn12,fnn21, mnn)
    im1seeds = np.array(im1seeds.cpu())
    im2seeds = np.array(im2seeds.cpu())
    seeds1 = np.array(k1[im1seeds].cpu())
    seeds2 = np.array(k2[im2seeds].cpu())

    # Delaunay triangulation
    tess = Delaunay(seeds1)
    dt_tr = tess.simplices
    # fig, ax = plt.subplots()
    # ax.margins(0.1)
    # ax.set_aspect('equal')
    plt.axis([0, im1shape[1], 0, im1shape[0]])
    cx, cy = zip(*seeds1)
    # ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tr), 'b.--')
    # plt.show()
    # fig2, ax2 = plt.subplots()
    # ax2.margins(0.1)
    # ax2.set_aspect('equal')
    # plt.axis([0, im2shape[1], 0, im2shape[0]])
    # sx, sy = zip(*seeds2)
    # ax2.triplot(matplotlib.tri.Triangulation(sx, sy, dt_tr), 'b.--')
    # plt.show()
    set_line = []
    for t in range(len(dt_tr)):
        not_in =True
        for k in range(len(set_line)):
            if (dt_tr[t][0] == set_line[k][0] and dt_tr[t][1] == set_line[k][1]) or (dt_tr[t][0] == set_line[k][1] and dt_tr[t][1] == set_line[k][0]):
                not_in = False
                break
            if (dt_tr[t][0] == set_line[k][0] and dt_tr[t][2] == set_line[k][1]) or (dt_tr[t][0] == set_line[k][1] and dt_tr[t][2] == set_line[k][0]):
                not_in = False
                break
            if (dt_tr[t][1] == set_line[k][0] and dt_tr[t][2] == set_line[k][1]) or (dt_tr[t][1] == set_line[k][1] and dt_tr[t][2] == set_line[k][0]):
                not_in = False
                break
        if not_in == True:
            set_line.append([dt_tr[t][0], dt_tr[t][1]])
            set_line.append([dt_tr[t][0], dt_tr[t][2]])
            set_line.append([dt_tr[t][1], dt_tr[t][2]])
    p_v_line = []
    cross_pt_num = []
    pt_list = []
    for j in range(len(seeds2)):
        pt_list.append(j)
        pt_line = []
        sum = 0
        for k in range(len(set_line)):   # Find the number of intersections of a line with a line in a line set
            if set_line[k][0] == j or set_line[k][1] == j:
                line = set_line[k]
                for t in set_line:
                    flag = isCross(seeds2[t[0]], seeds2[t[1]], seeds2[line[0]], seeds2[line[1]])
                    if flag == True:
                        sum += 1
                pt_line.append(line)
        p_v_line.append(pt_line)
        if len(pt_line) != 0:
            cross_pt_num.append(sum/len(pt_line))
        else:
            cross_pt_num.append(sum)
    ave_cross_num = np.mean(cross_pt_num)
    w_list = np.hstack((np.array(pt_list).reshape(-1, 1), np.array(cross_pt_num).reshape(-1, 1)))
    w_list = np.hstack((w_list, np.array(p_v_line).reshape(-1, 1)))
    w_list = np.array(sorted(w_list, key=lambda x: x[1], reverse=True))
    f = False
    while f == False:
        del_pt = w_list[0][0]
        w_list = w_list[1:]
        if w_list[0][1] <= ave_cross_num:
            f = True

    d_seeds1 = []
    d_seeds2 = []
    for i in range(len(w_list)):
        d_seeds1.append(im1seeds[int(w_list[i][0])])
        d_seeds2.append(im2seeds[int(w_list[i][0])])
    d_seeds1 = torch.from_numpy(np.array(d_seeds1))
    d_seeds2 = torch.from_numpy(np.array(d_seeds2))




    neigh_simi_seed = neigh_simi[d_seeds1]
    ave = torch.mean(neigh_simi_seed)

    # Adaptive Neighborhood Radius
    coff = [math.exp((ave - simi)) for simi in neigh_simi_seed]

    R1 = torch.Tensor(coff * (R1.repeat(len(d_seeds1))))
    R2 = torch.Tensor(coff * (R2.repeat(len(d_seeds2))))

    # Find the neighboring and coherent keyopints consistent with each seed
    local_neighs_mask, rdims, d_seeds1, d_seeds2, R1, R2 = extract_neighborhood_sets(
        o1, o2, s1, s2, dist1, d_seeds1, d_seeds2, k1, k2, R1, R2, fnn12,
        ORIENTATION_THR, SCALE_RATE_THR, SEARCH_EXP, MIN_INLIERS)

    if rdims.shape[0] == 0:
        # No seed point survived. Just output ratio-test matches. This should happen very rarely.
        absolute_im1idx = torch.where(scores < 0.8**2)[0]
        absolute_im2idx = fnn12[absolute_im1idx]
        return torch.stack([absolute_im1idx, absolute_im2idx], dim=1)

    # Format neighborhoods for parallel RANSACs
    im1loc, im2loc, ransidx, tokp1, tokp2 = extract_local_patterns(
        fnn12, local_neighs_mask, k1, k2, d_seeds1, d_seeds2, scores,R1,R2,SEARCH_EXP)
    ave_num = len(im1loc) / len(im1seeds)
    # Run the parallel confidence-based RANSACs to perform local affine verification
    inlier_idx, _, \
    inl_confidence, inlier_counts = ransac(ave_num=ave_num,xsamples=im1loc,
                                           ysamples=im2loc,
                                           rdims=rdims, iters=RANSAC_ITERS,
                                           refit=REFIT, config=config)

    conf = inl_confidence[ransidx[inlier_idx]]
    cnt = inlier_counts[ransidx[inlier_idx]].float()
    passed_inliers_mask = (conf >= MIN_CONF) & (cnt * (1 - 1/conf) >= MIN_INLIERS)
    accepted_inliers = inlier_idx[passed_inliers_mask]

    absolute_im1idx = tokp1[accepted_inliers]
    absolute_im2idx = tokp2[accepted_inliers]

    final_matches = torch.stack([absolute_im1idx, absolute_im2idx], dim=1)
    if final_matches.shape[0] > 1:
        return torch.unique(final_matches, dim=0)
    return final_matches
