import subprocess
import tqdm


with open('/tmp/chairs.txt', 'r') as f:
    chairs = f.readlines()

pbar = tqdm.tqdm(chairs)
for chair in pbar:
    chair = chair.strip()
    if chair.startswith('#'):
        continue
    pbar.set_description("Processing %s" % chair)

    # subprocess.run(['python', 'convert_to_neus.py', '--instance-id', chair])
    subprocess.run([
        'python', 'launch.py', '--config', 'configs/neus-dtu.yaml', '--gpu', '1', '--train',
        'dataset.cameras_file=cameras.npz',
        f'dataset.root_dir=/home/gruvi-3dv/workspace/datasets/ShapeNet-SRN/srn_chairs/chairs_train/chairs_2.0_train/{chair}',
        'tag="ShapeNet_SRN"',
        'model.geometry.isosurface.resolution=128',
        'model.radius=0.6',
        'model.learned_background=True',
        'system.loss.lambda_mask=0.1',
        'trainer.max_steps=10000',
    ])
