
from __future__ import annotations

from adaptive_evm import Params, adaptive_dynamic_weak_texture_amplification


def main() -> None:
    input_video = "EP08_04.avi"
    output_video = "enhanced_EP19_03f.avi"

    params = Params(
        tau=8,
        low_freq=0.5,
        high_freq=5.0,
        num_levels=4,
        target_r=0.11,
        alpha0=50,
        alpha_min=1,
        alpha_max=80,
        beta=0.1,
        max_iter=100,
        tol=1e-3,
        spatial_high_freq_threshold=15,
        k=1,
        epsilon=1e-8,
    )

    _, alpha_opt, log_table = adaptive_dynamic_weak_texture_amplification(
        input_video, output_video, params
    )
    log_table.to_csv("amplification_log.csv", index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    main()


# def main() -> None:
#     input_video = "EP08_04.avi"
#     output_video = "enhanced_EP19_03f.avi"

#     params = Params(
#         tau=8,
#         low_freq=0.5,
#         high_freq=5.0,
#         num_levels=4,
#         target_r=0.15,
#         alpha0=50,
#         alpha_min=1,
#         alpha_max=80,
#         beta=0.2,
#         max_iter=100,
#         tol=1e-3,
#         spatial_high_freq_threshold=15,
#         k=1,
#         epsilon=1e-8,
#     )

#     _, alpha_opt, log_table = adaptive_dynamic_weak_texture_amplification(
#         input_video, output_video, params
#     )
#     log_table.to_csv("amplification_log.csv", index=False, encoding="utf-8-sig")



# if __name__ == "__main__":
#     main()



# if __name__ == "__main__":
#     main()


# def main() -> None:
#     input_video = "EP08_04.avi"
#     output_video = "enhanced_EP19_03f.avi"

#     params = Params(
#         tau=8,
#         low_freq=0.5,
#         high_freq=5.0,
#         num_levels=4,
#         target_r=0.15,
#         alpha0=50,
#         alpha_min=1,
#         alpha_max=80,
#         beta=0.3,
#         max_iter=100,
#         tol=1e-3,
#         spatial_high_freq_threshold=15,
#         k=1,
#         epsilon=1e-8,
#     )

#     _, alpha_opt, log_table = adaptive_dynamic_weak_texture_amplification(
#         input_video, output_video, params
#     )
#     log_table.to_csv("amplification_log.csv", index=False, encoding="utf-8-sig")



# if __name__ == "__main__":
#     main()
