import configargparse
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    parser.add_argument("--multi_gpu", action='store_true', help='use multiple gpu on the server')
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='../logs', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')
    parser.add_argument("--places365_model_path", type=str, default='', help='ckpt path of places365 pretrained model')

    # 7Scenes
    parser.add_argument("--trainskip", type=int, default=1, help='will load 1/N images from train sets, useful for large datasets like 7 Scenes')
    parser.add_argument("--df", type=float, default=1., help='image downscale factor')
    parser.add_argument("--reduce_embedding", type=int, default=-1, help='fourier embedding mode: -1: paper default, \
                                                                        0: reduce by half, 1: remove embedding, 2: DNeRF embedding')
    parser.add_argument("--epochToMaxFreq", type=int, default=-1, help='DNeRF embedding mode: (based on Nerfie paper): \
                                                                        hyper-parameter for when α should reach the maximum number of frequencies m')
    parser.add_argument("--render_pose_only", action='store_true', help='render a spiral video for 7 Scene')
    parser.add_argument("--save_pose_avg_stats", action='store_true', help='save a pose avg stats to unify NeRF, posenet, direct-pn training')
    parser.add_argument("--load_pose_avg_stats", action='store_true', help='load precomputed pose avg stats to unify NeRF, posenet, nerf tracking training')
    parser.add_argument("--train_local_nerf", type=int, default=-1, help='train local NeRF with ith training sequence only, ie. Stairs can pick 0~3')
    parser.add_argument("--render_video_train", action='store_true', help='render train set NeRF and save as video, make sure render_test is True')
    parser.add_argument("--render_video_test", action='store_true', help='render val set NeRF and save as video,  make sure render_test is True')
    parser.add_argument("--frustum_overlap_th", type=float, help='frustsum overlap threshold')
    parser.add_argument("--no_DNeRF_viewdir", action='store_true', default=False, help='will not use DNeRF in viewdir encoding')
    parser.add_argument("--load_unique_view_stats", action='store_true', help='load unique views frame index')
    parser.add_argument("--finetune_unlabel", action='store_true', help='finetune unlabeled sequence like MapNet')
    parser.add_argument("--i_eval",   type=int, default=20, help='frequency of eval posenet result')
    parser.add_argument("--save_all_ckpt", action='store_true', help='save all ckpts for each epoch')
    parser.add_argument("--val_on_psnr", action='store_true', default=False, help='EarlyStopping with max validation psnr')

    # NeRF training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1536, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', default=True, help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')

    # NeRF-Hist training options
    parser.add_argument("--NeRFH", action='store_true', default=True, help='new implementation for NeRFH, please add --encode_hist')
    parser.add_argument("--N_vocab", type=int, default=1000,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument("--fix_index", action='store_true', help='fix training frame index as 0')
    parser.add_argument("--encode_hist", default=False, action='store_true', help='encode histogram instead of frame index')
    parser.add_argument("--hist_bin", type=int, default=10, help='image histogram bin size')
    parser.add_argument("--in_channels_a", type=int, default=50, help='appearance embedding dimension, hist_bin*N_a when embedding histogram')
    parser.add_argument("--in_channels_t", type=int, default=20, help='transient embedding dimension, hist_bin*N_tau when embedding histogram')
    parser.add_argument("--svd_reg", default=False, action='store_true', help='use svd regularize output at training')

    # NeRF rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True, help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--no_grad_update", action='store_true', default=False, help='do not update nerf in training')
    parser.add_argument("--tinyimg", action='store_true', default=False, help='render nerf img in a tiny scale image, this is a temporal compromise for direct feature matching, must FIX later')
    parser.add_argument("--tinyscale", type=float, default=4., help='determine the scale of downsizing nerf rendring, must FIX later')

    ##################### PoseNet Settings ########################
    parser.add_argument("--pose_only", type=int, default=1, help='posenet type to train, \
                        1: train baseline posenet, 2: posenet+nerf manual optimize, \
                        3: VLocNet, 4: DGRNet')
    parser.add_argument("--learning_rate", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=1, help='dataloader batch size, Attention: this is NOT the actual training batch size, \
                        please use --featurenet_batch_size for training')
    parser.add_argument("--featurenet_batch_size", type=int, default=8, help='featurenet training batch size, choose smaller batch size')
    parser.add_argument("--pretrain_model_path", type=str, default='', help='model path of pretrained model')
    parser.add_argument("--model_name", type=str, help='pose model output folder name')
    parser.add_argument("--combine_loss_w", nargs='+', type=float, default=[1, 1, 1], 
                        help='weights of combined loss ex, [1, 1, 1], \
                        default None, only use when combine_loss is True')
    parser.add_argument("--patience", nargs='+', type=int, default=[200, 50], help='set training schedule for patience [EarlyStopping, reduceLR]')
    parser.add_argument("--resize_factor", type=int, default=2, help='image resize downsample ratio')
    parser.add_argument("--freezeBN", action='store_true', help='Freeze the Batch Norm layer at training PoseNet')
    parser.add_argument("--preprocess_ImgNet", action='store_true', help='Normalize input data for PoseNet')
    parser.add_argument("--eval", action='store_true', help='eval model')
    parser.add_argument("--no_save_multiple", action='store_true', help='default, save multiple posenet model, if true, save only one posenet model')
    parser.add_argument("--resnet34", action='store_true', default=False, help='use resnet34 backbone instead of mobilenetV2')
    parser.add_argument("--efficientnet", action='store_true', default=False, help='use efficientnet-b3 backbone instead of mobilenetV2')
    parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate for resnet34 backbone')
    parser.add_argument("--DFNet", action='store_true', default=False, help='use DFNet')
    parser.add_argument("--DFNet_s", action='store_true', default=False, help='use accelerated DFNet, performance is similar to DFNet but slightly faster')
    parser.add_argument("--featurelossonly", action='store_true', default=False, help='only use feature loss to train feature extraction')
    parser.add_argument("--random_view_synthesis", action='store_true', default=False, help='add random view synthesis')
    parser.add_argument("--rvs_refresh_rate", type=int, default=2, help='re-synthesis new views per X epochs')
    parser.add_argument("--rvs_trans", type=float, default=5, help='jitter range for rvs on translation')
    parser.add_argument("--rvs_rotation", type=float, default=1.2, help='jitter range for rvs on rotation, this is in log_10 uniform range, log(15) = 1.2')
    parser.add_argument("--d_max", type=float, default=1, help='rvs bounds d_max')
    parser.add_argument("--val_batch_size", type=int, default=1, help='batch_size for validation, higher number leads to faster speed')

    # legacy mesh options
    parser.add_argument("--mesh_only", action='store_true', help='do not optimize, reload weights and save mesh to a file')
    parser.add_argument("--mesh_grid_size", type=int, default=80,help='number of grid points to sample in each dimension for marching cubes')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--epochs", type=int, default=2000,help='number of epochs to train')
    parser.add_argument("--poselossonly", action='store_true', help='eval model')
    parser.add_argument("--tripletloss", action='store_true', help='use triplet loss at training featurenet, this is to prevent catastophic failing')
    parser.add_argument("--triplet_margin", type=float,default=1., help='triplet loss margin hyperparameter')
    parser.add_argument("--render_feature_only", action='store_true', default=False, help='render features and save to a path')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / 7Scenes')
    parser.add_argument("--testskip", type=int, default=1, help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## legacy blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,  help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--no_bd_factor", action='store_true', default=False, help='do not use bd factor')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=200, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=200, help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, help='frequency of render_poses video saving')

    return parser