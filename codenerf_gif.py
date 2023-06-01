import sys
from pathlib import Path
import cv2
import imageio


if __name__ == '__main__':
    path = Path(sys.argv[1])

    imgs = []
    for i in range(200):
        img = cv2.imread(str(Path(path / f'opt1_{i}.png')))[:, :224, ::-1]
        img = cv2.putText(
            img.copy(), str(i), (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0, 255), 1, cv2.LINE_AA
        )
        imgs.append(img)
    imageio.mimsave(f'/tmp/{path.name}_opt.gif', imgs, fps=30)

    imgs = []
    for i in range(1, 100):
        img = cv2.imread(str(Path(path / f'{i}_1.png')))[:, :224, ::-1]
        imgs.append(img)
    imageio.mimsave(f'/tmp/{path.name}_pred.gif', imgs, fps=15)
