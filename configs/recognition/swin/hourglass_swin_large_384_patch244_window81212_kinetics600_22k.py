_base_ = [
    './swin_large_384_patch244_window81212_kinetics600_22k.py'
]
model=dict(
    backbone=dict(
        type='HourglassSwinTransformer3D',
        patch_size=(2,4,4), 
        window_size=(8,12,12), 
        drop_path_rate=0.5,
        clustering_location=12,
        token_clustering_cfg=dict(
            clustering_shape=(8, 6, 6),
            n_iters=5,
            temperature=0.02,
            window_size=5,
        ),
        token_reconstruction_cfg=dict(
            k=25,
            temperature=0.02,
        ),
    ), 
    test_cfg=dict(max_testing_views=1))

