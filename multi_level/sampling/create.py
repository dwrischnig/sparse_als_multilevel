from affine_equation import AffineEquation
import argparse

def main():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("-L", "--L", type=int, default=5, help="maximum level (default: 5)")
    parser.add_argument("-nx", "--min_nx", type=int, default=20, help="number of cells per axis for level 1 (default: 20)")
    parser.add_argument("-tss", "--test_set_size", type=int, default=500, help="Size of test set (default: 500)")
    
    args = parser.parse_args()

    eq = AffineEquation()

    L = args.L
    eq.initialise_spaces(L, min_nx = args.min_nx)

    eq.generate_test_set(args.test_set_size)
    eq.save_results(filename_test = f"test_L{L}_nx{args.min_nx}")

    while L > 0:
        eq.perform_multilevel_procedure()
        eq.save_results(filename_multilevel = f"ml_L{L}_nx{args.min_nx}")

        eq.perform_singlelevel_procedure()
        eq.save_results(filename_singlelevel = f"sl_L{L}_nx{args.min_nx}")
        
        eq.initialise_spaces(L, min_nx = args.min_nx)

        L -= 1

if __name__ == "__main__":
    main()

