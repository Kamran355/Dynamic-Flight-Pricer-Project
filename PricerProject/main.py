import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils import print_header, print_warning


def menu() -> str:
    print_header("FLIGHT PRICING ENGINE  —  MAIN MENU")
    print()
    print("  [1]  Price a new flight")
    print("  [2]  View analytics & model diagnostics report")
    print("  [3]  Exit")
    print()
    while True:
        choice = input("  Select (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print_warning("Please enter 1, 2, or 3.")


def main():
    while True:
        choice = menu()
        if choice == "1":
            from pricer import run_pricing_session
            run_pricing_session()
        elif choice == "2":
            from reports import run_report
            run_report()
        elif choice == "3":
            print("\n  Safe skies.\n")
            break
        print()
        input("  Press Enter to return to the main menu...")


if __name__ == "__main__":
    main()