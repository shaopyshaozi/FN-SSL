import os
import numpy as np
import soundfile as sf
from RecordData_multiple import RealData
from tqdm import tqdm


def generate_fixed_set(
    save_dir,
    data_dir,
    target_csvs,
    noise_dir,
    use_mic_id=[1, 3, 5, 7, 0],
    max_source=2,
):
    os.makedirs(save_dir, exist_ok=True)

    dataset = RealData(
        data_dir=data_dir,
        target_dir=target_csvs,
        noise_dir=noise_dir,
        use_mic_id=use_mic_id,
        max_source=max_source,
        on_the_fly=True,
    )

    n = len(dataset)
    print(f"Generating {n} samples into {save_dir}")

    for idx in tqdm(range(n)):
        # fixed seed per index -> deterministic generation
        mic, targets, vad, array_topo, distances = dataset[(idx, idx)]

        base = f"sample_{idx:06d}"
        wav_path = os.path.join(save_dir, f"{base}.wav")
        targets_path = os.path.join(save_dir, f"targets_{base}.npy")
        dis_path = os.path.join(save_dir, f"dis_{base}.npy")
        vad_path = os.path.join(save_dir, f"vad_{base}.npy")

        sf.write(wav_path, mic, 16000)
        np.save(targets_path, targets.numpy())
        np.save(dis_path, distances.numpy())
        np.save(vad_path, vad.numpy())

        if idx % 100 == 0:
            print(f"Saved {idx}/{n}")

    print(f"Done. Generated {n} samples in {save_dir}")


if __name__ == "__main__":
    # Validation set
    generate_fixed_set(
        save_dir=r"E:/RealMAN/val_gen_3spk",
        data_dir=r"E:/RealMAN/",
        target_csvs=[
            r"E:/RealMAN/val_raw/val_static_source_location.csv",
            r"E:/RealMAN/val_raw/val_moving_source_location.csv",
        ],
        noise_dir=r"E:/RealMAN/val_raw/ma_noise",
        use_mic_id=[2,4,6,8],
        max_source=3,
    )

    # # Test set
    # generate_fixed_set(
    #     save_dir=r"E:/RealMAN/test_gen",
    #     data_dir=r"E:/RealMAN/",
    #     target_csvs=[
    #         r"E:/RealMAN/test_raw/test_static_source_location.csv",
    #         r"E:/RealMAN/test_raw/test_moving_source_location.csv",
    #     ],
    #     noise_dir=r"E:/RealMAN/test_raw/ma_noise",
    #     use_mic_id=[2,4,6,8],
    #     max_source=3,
    # )