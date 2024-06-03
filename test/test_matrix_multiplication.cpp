#include "matrix_multiplication.h"
#include <gtest/gtest.h>
#include <random>

#define FUZZY_IT 50

// CORRECT IMPLEMENTATION OF MATRIX MULTIPLICATION (FOR CROSS-CHECKS)

void multiplyMatricesWithoutErrors(const std::vector<std::vector<int>> &A,
                                   const std::vector<std::vector<int>> &B,
                                   std::vector<std::vector<int>> &C, int rowsA, int colsA,
                                   int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// TESTS ON UNITARY-DIMENSION MATRICES ********************************************************
// The following suite of tests wants to check various behaviour when the function is
// required to perform a simple scalar multiplication


/*
 * The following test checks the behaviour of the function with combination of signed operation
 */
TEST(UnitaryMatricesTests, SignsTest) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> AP = {
            {1}
    };
    std::vector<std::vector<int>> BP = {
            {1}
    };
    std::vector<std::vector<int>> AN = {
            {1}
    };
    std::vector<std::vector<int>> BN = {
            {1}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AP, BP, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AP, BP, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Sign Test failed on (+) * (+)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AP, BN, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AP, BN, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Sign Test failed on (+) * (-)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AN, BP, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AN, BP, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Sign Test failed on (-) * (+)";

    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AN, BN, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AN, BN, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Sign Test failed on (-) * (-)";
}

/*
 * The following test checks the behaviour of the function when at least one of the factors is equal to 0
 */
TEST(UnitaryMatricesTests, ZeroTest) {
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {0}
    };
    std::vector<std::vector<int>> B = {
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 1));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Zero Test failed on (0) * (1)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(B, A, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(B, A, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Zero Test failed on (1) * (0)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(A, A, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(A, A, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Zero Test failed on (0) * (0)";
}

/*
 * The following test does fuzzy test on the behaviour of the function with random value
 */
TEST(UnitaryMatricesTests, FuzzyTest){
    int aRows = 1;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {{0}};
    std::vector<std::vector<int>> B = {{0}};

    std::random_device rd;
    for (int i = 0; i < FUZZY_IT; i++) {
        std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
        std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-200, 200);
        A[0][0] = dis(gen);
        B[0][0] = dis(gen);

        multiplyMatrices(A, B, C, aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

        EXPECT_EQ(C,expected) << "Fuzzy test iteration failed";
    }
}


// TEST ON SQUARE MATRICES ************************************************************************************
// The following suite of tests wants to check various behaviour when the function is
// required to perform matrix multiplication on the "smallest" matrix first factor format (i.e. 2x2)

/*
 * The following test checks the behaviour of the function with signed vector
 */
TEST(SquareMatricesTests, SignTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {8,  8},
            {-3, 7}
    };
    std::vector<std::vector<int>> B = {
            {-1},
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);

    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "2x2 Sign test failed";
}

/*
 * The following test checks the behaviour of the function when at least one of the factors is equal to 0
 */
TEST(SquareMatricesTests, ZeroTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> AZ = {
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> BU = {
            {1},
            {1}
    };

    std::vector<std::vector<int>> AU = {
            {1, 1},
            {1, 1}
    };
    std::vector<std::vector<int>> BZ = {
            {0},
            {0}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 1));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BU, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BU, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "2x2 Zero Test failed on (0) * (1)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AU, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AU, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "2x2 Zero Test failed on (1) * (0)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "2x2 Zero Test failed on (0) * (0)";
}

/*
 * The following test checks the behaviour of the function when matrix is an identity or vector is an identity
 */
TEST(SquareMatricesTests, IdentityTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> AI = {
            {1, 0},
            {0, 1}
    };
    std::vector<std::vector<int>> BN = {
            {5},
            {5}
    };

    std::vector<std::vector<int>> AN = {
            {1,  2},
            {-3, 7}
    };
    std::vector<std::vector<int>> BI = {
            {1},
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AI, BN, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AI, BN, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "2x2 Matrix Identity  Test failed";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AN, BI, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AN, BI, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "2x2 Vector Identity failed";
}

/*
 * The following test does fuzzy test on the behaviour of the function with random values
 */
TEST(SquareMatricesTests, FuzzyTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 1;

    std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols,0));
    std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols,0));

    std::random_device rd;
    for (int i = 0; i < FUZZY_IT; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 9);
        std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
        std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));
        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, C, aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);
        EXPECT_EQ(C, expected) << "2x2 Fuzzy Test iteration failed";
    }
}


// TEST ON RECTANGULAR MATRICES ********************************************************
// The following suite of tests wants to check various behaviour when the function is
// required to perform matrix multiplication with rectangular form first factor

/*
 * The following test checks the behaviour of the function when at least one of the factors is equal to 0
 */
TEST(RectangularMatricesTests, ZeroTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;

    std::vector<std::vector<int>> AZ = {
            {0, 0, 0},
            {0, 0, 0}
    };
    std::vector<std::vector<int>> AU = {
            {1, 2, 3},
            {4, 5, 6}
    };
    std::vector<std::vector<int>> BZ = {
            {0, 0},
            {0, 0},
            {0, 0}
    };
    std::vector<std::vector<int>> BU = {
            {1, 2},
            {3, 4},
            {5, 6}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 1));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BU, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BU, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Rect Zero Test failed on (0) * (1)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AU, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AU, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Rect Zero Test failed on (1) * (0)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Rect Zero Test failed on (0) * (0)";
}

/*
 * The following test checks the behaviour of the function when factors contain signed values
 */
TEST(RectangularVector, SignTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {-33, -33, -33},
            {15, 10,  0}
    };
    std::vector<std::vector<int>> B = {
            {-1},
            {-1},
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {6},
            {15}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Rect Sign Test failed";
}

/*
 * The following test checks the behaviour of the function when at least one of the factors is an identity
 */
TEST(RectangularMatricesTests, IdentityTest) {
    int aRows = 2;
    int aCols = 2;
    int bCols = 2;

    std::vector<std::vector<int>> AI = {
            {1, 0},
            {0, 1}
    };
    std::vector<std::vector<int>> BN = {
            {17, 6},
            {7,  1},
            {8,  2}
    };
    std::vector<std::vector<int>> AN = {
            {6, 6},
            {5, 5}
    };
    std::vector<std::vector<int>> BI = {
            {1, 0},
            {0, 1},
            {0, 0}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AI, BN, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AI, BN, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Rect Matrix Identity Test failed";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AN, BI, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AN, BI, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Rect Vector Identity test failed";
}

/*
 * The following test checks the behaviour of the function when the second factor is a vector
 */
TEST(RectangularMatricesTest, NonRectVectorTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {6, 6, 6},
            {5, 5, 5}
    };
    std::vector<std::vector<int>> B = {
            {1},
            {1},
            {1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    std::vector<std::vector<int>> expected = {
            {6},
            {15}
    };
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Rect Vect-Second-Factor test failed";
}

/*
 * The following test does fuzzy test on the behaviour of the function with random values
 */
TEST(RectangularMatricesTests, FuzzyTest) {
    int aRows = 2;
    int aCols = 3;
    int bCols = 2;

    std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols,0));
    std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols,0));

    std::random_device rd;
    for (int i = 0; i < FUZZY_IT; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 9);
        std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
        std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));
        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, C, aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);
        EXPECT_EQ(C, expected) << "Rect Fuzzy Test iteration failed";
    }
}


// TEST ON VECTORS MATRICES ********************************************************
// The following suite of tests wants to check various behaviour when the function is
// required to perform matrix multiplication with vector form factors

/*
 * The following test checks the behaviour of the function when one of the factor is a unitary vector
 */
TEST(VectorMatricesTests, UnitatyTest) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> AN = {
            {33, 33, 33}
    };
    std::vector<std::vector<int>> BI = {
            {1},
            {1},
            {1}
    };
    std::vector<std::vector<int>> AI = {
            {1, 1, 1}
    };
    std::vector<std::vector<int>> BN = {
            {22},
            {17},
            {15}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AI, BN, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AI, BN, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Vect First Unitary Test failed";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AN, BI, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AN, BI, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Vect Second Unitary test failed";
}

/*
 * The following test checks the behaviour of the function when factors contains signed values
 */
TEST(VectorMatricesTests, SignTest) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {33, 33, 33}
    };
    std::vector<std::vector<int>> B = {
            {-1},
            {-1},
            {-1}
    };
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "Vect Sign test failed";
}

/*
 * The following test checks the behaviour of the function at least one of factor is zero matrix
 */
TEST(VectorMatricesTests, ZeroTest) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> AZ = {
            {0, 0, 0}
    };
    std::vector<std::vector<int>> AU = {
            {1, 1, 1}
    };
    std::vector<std::vector<int>> BZ = {
            {0},
            {0},
            {0}
    };
    std::vector<std::vector<int>> BU = {
            {1},
            {1},
            {1}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 1));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BU, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BU, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Vect Zero Test failed on (0) * (1)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AU, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AU, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Vect Zero Test failed on (1) * (0)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "Vect Zero Test failed on (0) * (0)";
}

/*
 * The following test does fuzzy test on the behaviour of the function with random values
 */
TEST(VectorMatricesTests, FuzzyTest) {
    int aRows = 1;
    int aCols = 3;
    int bCols = 1;

    std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols, 0));
    std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols, 0));

    std::random_device rd;
    for (int i = 0; i < FUZZY_IT; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 9);
        std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
        std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));
        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, C, aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);
        EXPECT_EQ(C, expected) << "Vect Fuzzy Test iteration failed";
    }
}

// TEST ON VECTOR-SCALAR MATRICES ********************************************************
// The following suite of tests wants to check various behaviour when the function is
// required to perform matrix multiplication on (vector x scalar) form

/*
 * The following test checks the behaviour of the function at least one of factor is zero matrix
 */
TEST(VectorScalarTests, ZeroTest) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> AZ = {
            {0},
            {0},
            {0},
            {0},
            {0},
            {0},
            {0}
    };
    std::vector<std::vector<int>> AU = {
            {1},
            {1},
            {1},
            {1},
            {1},
            {1},
            {1}
    };
    std::vector<std::vector<int>> BU = {
            {1}
    };
    std::vector<std::vector<int>> BZ = {
            {0}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 1));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BU, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BU, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "VectScalar Zero Test failed on (0) * (1)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AU, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AU, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "VectScalar Zero Test failed on (1) * (0)";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 1));

    multiplyMatrices(AZ, BZ, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AZ, BZ, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "VectScalar Zero Test failed on (0) * (0)";
}

/*
 * The following test checks the behaviour of the function when one of the factor is a unitary vector
 */
TEST(VectorScalarTests, UnitaryTest) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> AI = {
            {1},
            {1},
            {1},
            {1},
            {1},
            {1},
            {1}
    };
    std::vector<std::vector<int>> AN = {
            {3},
            {3},
            {3},
            {3},
            {3},
            {3},
            {3}
    };
    std::vector<std::vector<int>> BI = {
            {1}
    };
    std::vector<std::vector<int>> BN = {
            {3}
    };
    
    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AI, BN, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AI, BN, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "VectScalar First Unitary Test failed";


    C = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));
    expected = std::vector<std::vector<int>>(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(AN, BI, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(AN, BI, expected, aRows, aCols, bCols);

    EXPECT_EQ(C, expected) << "VectScalar Second Unitary test failed";
}

/*
 * The following test checks the behaviour of the function when factors contains signed values
 */
TEST(VectorScalarTests, SignTest) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A = {
            {-1},
            {1},
            {1},
            {-1},
            {1},
            {-1},
            {1}
    };
    std::vector<std::vector<int>> B = {
            {65}
    };

    std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A, B, C, aRows, aCols, bCols);
    multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);

    ASSERT_EQ(C, expected) << "VectScalar Sign test failed";
}

/*
 * The following test does fuzzy test on the behaviour of the function with random values
 */
TEST(VectorScalarTests, FuzzyTest) {
    int aRows = 7;
    int aCols = 1;
    int bCols = 1;

    std::vector<std::vector<int>> A(aRows, std::vector<int>(aRows,0));
    std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols, 0));

    std::random_device rd;
    for (int i = 0; i < FUZZY_IT; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 9);
        std::vector<std::vector<int>> C(aRows, std::vector<int>(aCols, 0));
        std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, C, aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);
        EXPECT_EQ(C, expected) << "VectScalar Fuzzy Test iteration failed";
    }
}

// TEST ON RANDOM MATRICES DIMENSIONS ********************************************************
// The following tests wants to check various behaviour when the function is
// required to perform matrix multiplication with matrix with random dimensions

TEST(RandomDimensionsTests, FuzzyTest) {
    int aRows, aCols, bCols;
    std::random_device rd;

    for (int i = 0; i < FUZZY_IT; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 9);
        aRows = dis(gen);
        aCols = dis(gen);
        bCols = dis(gen);
        std::vector<std::vector<int>> A(aRows, std::vector<int>(aCols, 0));
        std::vector<std::vector<int>> B(aCols, std::vector<int>(bCols, 0));
        std::vector<std::vector<int>> C(aRows, std::vector<int>(bCols, 0));
        std::vector<std::vector<int>> expected(aRows, std::vector<int>(bCols, 0));

        for (int j = 0; j < aRows; j++) {
            for (int k = 0; k < aCols; k++) {
                A[j][k] = dis(gen);
            }
        }
        for (int j = 0; j < aCols; j++) {
            for (int k = 0; k < bCols; k++) {
                B[j][k] = dis(gen);
            }
        }
        multiplyMatrices(A, B, C, aRows, aCols, bCols);
        multiplyMatricesWithoutErrors(A, B, expected, aRows, aCols, bCols);
        EXPECT_EQ(C, expected) << "Random dimension matrices fuzzy test iteration failed";
    }
}

// TEST ON MATRICES PROPERTIES ********************************************************
// The following tests wants to check if the result of the function respect the
// properties of matrix multiplication
//
// Note that Neutral element and Zero element has been already tested multiple times
// in previous tests

/*
 * This test checks the commutative property (that matrix multiplication does not have)
 */
TEST(MatricesProperties, NotCommutative){
    int aRows = 2;
    int aCols = 2;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {10,15},
            {60,80},
    };
    std::vector<std::vector<int>> B = {
            {1,15},
            {9,8}
    };

    std::vector<std::vector<int>> C1(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> C2(aRows, std::vector<int>(bCols, 0));


    multiplyMatrices(A, B, C1, aRows, aCols, bCols);
    multiplyMatrices(B, A, C2, aRows, aCols, bCols);

    ASSERT_NE(C1,C2) << "Commutative test failed";
}

/*
 * This test checks the associative property (that matrix multiplication have)
 */
TEST(MatricesProperties, Associative){
    int aRows = 2;
    int aCols = 2;
    int bCols = 2;

    std::vector<std::vector<int>> A = {
            {10,15},
            {60,80},
    };
    std::vector<std::vector<int>> B = {
            {1,15},
            {9,8}
    };
    std::vector<std::vector<int>> C = {
            {8,40},
            {0,20000}
    };

    std::vector<std::vector<int>> T1(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> T2(aRows, std::vector<int>(bCols, 0));

    std::vector<std::vector<int>> D1(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> D2(aRows, std::vector<int>(bCols, 0));


    multiplyMatrices(A, B, T1, aRows, aCols, bCols);
    multiplyMatrices(T1, C, D1, aRows, aCols, bCols);

    multiplyMatrices(B, C, T2, aRows, aCols, bCols);
    multiplyMatrices(A, T2, D2, aRows, aCols, bCols);

    ASSERT_EQ(D1,D2) << "Associative test failed";
}

/*
 * This test checks distributive property (that matrix multiplication have)
 */
TEST(MatricesProperties, Distributive){
    int aRows = 2;
    int aCols = 2;
    int bCols = aRows;

    std::vector<std::vector<int>> A = {
            {10,15},
            {60,80},
    };
    std::vector<std::vector<int>> B = {
            {1,15},
            {9,8}
    };
    std::vector<std::vector<int>> C = {
            {8,40},
            {0,20000}
    };



    std::vector<std::vector<int>> AC(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> BC(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> AC_BC(aRows, std::vector<int>(bCols, 0));

    multiplyMatrices(A,C, AC, aRows, aCols, bCols);
    multiplyMatrices(B,C, BC, aRows, aCols, bCols);

    for(int i=0; i<aRows; ++i){
        for(int j=0; j<aCols; ++j){
            AC_BC[i][j] = AC[i][j] + BC[i][j];
        }
    }

    std::vector<std::vector<int>> A_B(aRows, std::vector<int>(bCols, 0));
    std::vector<std::vector<int>> A_B_C(aRows, std::vector<int>(bCols, 0));

    for(int i=0; i<aRows; ++i){
        for(int j=0; j<aCols; ++j){
            A_B[i][j] = A[i][j] + B[i][j];
        }
    }

    multiplyMatrices(A_B, C, A_B_C, aRows, aCols, bCols);

    ASSERT_EQ(AC_BC,A_B_C) << "Distributive test failed";
}

// *********************************************************************************

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
