import pathlib

hops_path = pathlib.Path(__file__).absolute().parent.parent.parent
print(hops_path)
import sys

sys.path.append(str(hops_path))

from hops.core.integration import HiIdx, IdxDict


def useHiIdx():
    print("+---------------------------------------+")
    print("|   simple single environment example   |")
    print("+---------------------------------------+")

    print("init zero index (single environment)")
    idx = HiIdx(n_list=[3])  # a simple index vector with 3 dimension
    print("  pretty string", idx)
    print("  binary", idx.to_bin())

    print("increase first dimension by one")
    idx[0][0] += 1  # modication directly on the data object
    # numpy array of numpy arrays with type int16
    print("  pretty string", idx)
    print("  binary", idx.to_bin())

    print("create a second HiIdx instance by copy constructor")
    idx2 = HiIdx.from_other(idx)
    print("  ", idx2)

    idx2[0][2] = 5
    idx2_bin = idx2.to_bin()

    print("create a third HiIdx instance from binary data")
    idx3 = HiIdx(n_list=[3], other_bin=idx2_bin)
    print("  ", idx3)

    print("+-------------------------------+")
    print("|   multi environment example   |")
    print("+-------------------------------+")

    print("init zero index (1st env dim 3, 2nd env dim 2)")
    idx = HiIdx(n_list=[3, 2])
    print("  pretty string", idx)

    idx[0][0] += 1
    idx[1][1] += 1
    print("  pretty string", idx)


def simplex_condition():
    print("+----------------------------------+")
    print("|   single env simplex condition   |")
    print("+----------------------------------+")
    n_list, kmax = 3, 3
    print("dims HiIdx", n_list)
    print("kmax", kmax)

    d = IdxDict(n_list=n_list)
    d.make_simplex(kmax=kmax)
    d.print_all_idx()
    print("number of HiIdx", d.num_idx())

    idx = HiIdx(n_list=n_list)
    idx._data[2] = 1
    print(f"example: index of {idx} is {d.get_idx(idx)}")

    print("+---------------------------------+")
    print("|   multi env simplex condition   |")
    print("+---------------------------------+")
    n_list, kmax = [3, 2], 2
    print("dims HiIdx", n_list)
    print("kmax", kmax)
    d = IdxDict(n_list=n_list)
    d.make_simplex(kmax=kmax)
    d.print_all_idx()
    print("number of HiIdx", d.num_idx())
    idx = HiIdx(n_list=n_list)
    idx._data[2] = 1
    print(f"example: index of {idx} is {d.get_idx(idx)}")


def use_case_in_HOPS():
    print("+----------------------------------+")
    print("|   single env simplex condition   |")
    print("|   how to generate the HOPS       |")
    print("+----------------------------------+")
    n_list, kmax = [3], 3
    print("dims HiIdx", n_list)
    print("kmax", kmax)

    d = IdxDict(n_list=n_list)
    d.make_simplex(kmax=kmax)

    print("go through all hi_idx obtained for some truncation scheme, here simplex")
    for c in d.idx_dict:
        hi_idx = HiIdx(n_list=n_list, other_bin=c)
        print("  at hi_idx", hi_idx, "(idx {})".format(d.get_idx(hi_idx)))
        print("    - coupling to lower tire")
        for i in range(n_list[0]):
            hi_idx2 = HiIdx.from_other(hi_idx)
            hi_idx2._data[i] -= 1
            hi_idx2_bin = hi_idx2.to_bin()
            if hi_idx2_bin in d.idx_dict:
                print(
                    "      coupling to {} (idx {})".format(
                        hi_idx2, d.idx_dict[hi_idx2_bin]
                    )
                )
            else:
                print("      possible index {} not in idx_dict".format(hi_idx2))

        print("    - coupling to higher tire")
        for i in range(n_list[0]):
            hi_idx2 = HiIdx.from_other(hi_idx)
            hi_idx2._data[i] += 1
            hi_idx2_bin = hi_idx2.to_bin()
            if hi_idx2_bin in d.idx_dict:
                print(
                    "      coupling to {} (idx {})".format(
                        hi_idx2, d.idx_dict[hi_idx2_bin]
                    )
                )
            else:
                print("      possible index {} not in idx_dict".format(hi_idx2))


useHiIdx()
simplex_condition()
use_case_in_HOPS()
