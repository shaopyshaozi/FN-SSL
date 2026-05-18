import numpy as np
from itertools import permutations


def angular_error_deg(pred, gt):
    diff = (pred - gt + 180) % 360 - 180
    return abs(diff)


def convert_pred_angle(pred):
    # Convert [-180, 0) to [180, 360]
    # Example: -10 -> 350, -90 -> 270, -180 -> 180
    if pred < 0:
        return 360+pred
    return pred


doa_est = np.load(r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet2\hidden96_fre128\0_doaest.npy")
doa_gt  = np.load(r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet2\hidden96_fre128\0_doagt.npy")
vad_gt  = np.load(r"D:\邵鹏远\UCL\博1\code\FN-SSL\IPDnet2\hidden96_fre128\0_vadgt.npy")

az_est = doa_est[:, :, 1, :]
az_gt  = doa_gt[:, :, 1, :]

B, T, S = az_est.shape
all_errors = []

for b in range(B):
    print(f"\n=== Batch {b} ===")

    for t in range(T):
        active_gt = [
            s for s in range(S)
            if vad_gt[b, t, s] > 0 and az_gt[b, t, s] != 0
        ]

        if len(active_gt) == 0:
            continue

        best_errors = None
        best_perm = None
        best_pred_adjs = None

        for perm in permutations(range(S), len(active_gt)):
            errors = []
            pred_adjs = []

            for pred_s, gt_s in zip(perm, active_gt):
                pred_raw = az_est[b, t, pred_s]
                pred_adj = convert_pred_angle(pred_raw)

                err = angular_error_deg(pred_adj, az_gt[b, t, gt_s])

                errors.append(err)
                pred_adjs.append(pred_adj)

            mean_err = np.mean(errors)

            if best_errors is None or mean_err < np.mean(best_errors):
                best_errors = errors
                best_perm = perm
                best_pred_adjs = pred_adjs

        for pred_s, gt_s, pred_adj, err in zip(
            best_perm, active_gt, best_pred_adjs, best_errors
        ):
            pred_raw = az_est[b, t, pred_s]
            gt_angle = az_gt[b, t, gt_s]

            all_errors.append(err)

            print(
                f"t={t:03d}: pred_src={pred_s}, gt_src={gt_s}, "
                f"pred_raw={pred_raw:.2f}, pred_adj={pred_adj:.2f}, "
                f"gt={gt_angle:.2f}, err={err:.2f}"
            )

all_errors = np.array(all_errors)

print("\n===== PERMUTATION-MATCHED ACTIVE-FRAME SUMMARY =====")

if len(all_errors) > 0:
    print(f"Mean angular error: {all_errors.mean():.2f} deg")
    print(f"Median angular error: {np.median(all_errors):.2f} deg")
    print(f"RMSE angular error: {np.sqrt(np.mean(all_errors ** 2)):.2f} deg")
else:
    print("No active frames found.")