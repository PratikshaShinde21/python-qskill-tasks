# ============================================
# TASK 3 - Matrix Operations Tool (NumPy)
# QSkill Internship - Slab 1
# ============================================

import numpy as np

def print_matrix(matrix, label="Matrix"):
    """Display matrix in a structured format."""
    print(f"\n  {label}:")
    print("  " + "-" * (len(matrix[0]) * 8 + 1))
    for row in matrix:
        print("  |", end="")
        for val in row:
            print(f"  {val:4.1f} ", end="")
        print("|")
    print("  " + "-" * (len(matrix[0]) * 8 + 1))


def get_matrix_input(name):
    """Get matrix input from user."""
    print(f"\n📥 Enter {name}:")
    rows = int(input("   Number of rows    : "))
    cols = int(input("   Number of columns : "))
    matrix = []
    print(f"   Enter {rows}x{cols} elements row by row (space separated):")
    for i in range(rows):
        while True:
            try:
                row = list(map(float, input(f"   Row {i+1}: ").split()))
                if len(row) != cols:
                    print(f"   ⚠ Enter exactly {cols} values!")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("   ⚠ Enter numbers only!")
    return np.array(matrix)


def show_result(result, label):
    """Show result with label."""
    if isinstance(result, np.ndarray):
        print_matrix(result.tolist(), label)
    else:
        print(f"\n  {label}: {result:.4f}")


def matrix_operations_menu():
    """Main interactive menu."""
    print("\n" + "=" * 55)
    print("   🔢 MATRIX OPERATIONS TOOL")
    print("   QSkill Internship — Slab 1, Task 3")
    print("=" * 55)

    while True:
        print("\n📋 MAIN MENU:")
        print("   1. Addition          (A + B)")
        print("   2. Subtraction       (A - B)")
        print("   3. Multiplication    (A × B)")
        print("   4. Transpose         (Aᵀ)")
        print("   5. Determinant       |A|")
        print("   6. Run Demo (auto sample matrices)")
        print("   0. Exit")
        print()
        choice = input("👉 Choose operation (0-6): ").strip()

        if choice == "0":
            print("\n👋 Thank you for using Matrix Operations Tool!\n")
            break

        elif choice == "6":
            # ── Demo Mode ───────────────────────────────────────────────────
            print("\n" + "=" * 55)
            print("   🎬 DEMO MODE — Sample Matrices")
            print("=" * 55)

            A = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=float)

            B = np.array([[9, 8, 7],
                          [6, 5, 4],
                          [3, 2, 1]], dtype=float)

            C = np.array([[2, 1],
                          [5, 3]], dtype=float)   # for determinant demo

            print_matrix(A.tolist(), "Matrix A (3×3)")
            print_matrix(B.tolist(), "Matrix B (3×3)")
            print_matrix(C.tolist(), "Matrix C (2×2)")

            # Addition
            print("\n➕ A + B:")
            show_result(A + B, "Result")

            # Subtraction
            print("\n➖ A - B:")
            show_result(A - B, "Result")

            # Multiplication
            print("\n✖  A × B:")
            show_result(A @ B, "Result")

            # Transpose
            print("\n🔄 Transpose of A:")
            show_result(A.T, "Aᵀ")

            # Determinant
            print("\n📐 Determinant of C:")
            det = np.linalg.det(C)
            show_result(det, "|C|")

            print("\n✅ Demo complete!")

        elif choice in ["1", "2", "3"]:
            A = get_matrix_input("Matrix A")
            B = get_matrix_input("Matrix B")

            if choice == "1":
                if A.shape != B.shape:
                    print("\n⚠  Addition requires same shape matrices!")
                    continue
                show_result(A + B, "A + B")

            elif choice == "2":
                if A.shape != B.shape:
                    print("\n⚠  Subtraction requires same shape matrices!")
                    continue
                show_result(A - B, "A - B")

            elif choice == "3":
                if A.shape[1] != B.shape[0]:
                    print(f"\n⚠  Multiplication not possible! "
                          f"A columns ({A.shape[1]}) ≠ B rows ({B.shape[0]})")
                    continue
                show_result(A @ B, "A × B")

        elif choice == "4":
            A = get_matrix_input("Matrix A")
            print_matrix(A.tolist(), "Original Matrix A")
            show_result(A.T, "Transpose Aᵀ")

        elif choice == "5":
            A = get_matrix_input("Square Matrix A")
            if A.shape[0] != A.shape[1]:
                print("\n⚠  Determinant requires a SQUARE matrix!")
                continue
            det = np.linalg.det(A)
            print_matrix(A.tolist(), "Matrix A")
            show_result(det, "Determinant |A|")
            if abs(det) < 1e-10:
                print("  ℹ Determinant ≈ 0 → Matrix is SINGULAR (no inverse)")
            else:
                print("  ℹ Matrix is NON-SINGULAR (inverse exists)")
        else:
            print("\n⚠  Invalid choice! Enter 0-6.")


# ── Run the Tool ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    matrix_operations_menu()
