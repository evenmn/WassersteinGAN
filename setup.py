from setuptools import setup

setup(name='wccdcgan',
      version='0.1',
      description='General Pytorch GAN wrapper',
      url='http://github.com/evenmn/wccdcgan',
      author='Even Marius Nordhagen',
      author_email='evenmn@mn.uio.no',
      license='MIT',
      packages=['wccdcgan', 'wccdcgan/models'],
      install_requires=["numpy", "tqdm", "torch", "torchvision", "pathlib", "matplotlib"],
      zip_safe=False)
