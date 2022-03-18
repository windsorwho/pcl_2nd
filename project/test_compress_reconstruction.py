import compress
import reconstruct


def test_full_process():
    compress.compress('128')
    reconstruct.reconstruct('128')


if __name__ == '__main__':
    test_full_process()
