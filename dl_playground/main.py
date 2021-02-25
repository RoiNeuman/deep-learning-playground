import os

from binary_classification.run import run

# Enable GPU use for Tensorflow
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def main():
    run()


if __name__ == '__main__':
    main()
