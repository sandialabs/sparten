//@HEADER
// ************************************************************************
//     SparTen: Software for Sparse Tensor Decompositions
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
// ************************************************************************
//@HEADER

#include <unit_test/CommandLineOptionsUnitTest.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace unitTest
{
    void CommandLineOptionsUnitTest::SetUp(){

        // test values by type
        val_uint64_t = 20;
        min_uint64_t = 1;
        max_uint64_t = 1000;
        val_double = 1e-10;
        min_double = 0.0;
        max_double = 1.0;

        /* Parameter defaults */
        _default_args = {
            "--init", "init.txt",
            "--output", "ktensor.txt",
            "--index-base", "1",
            "--method", "Multiplicative-Update",
            "--rank", "5",
            "--seed", "0",
            "--maxiters", "20",
            "--maxinneriters", "10",
            "--tol", "1e-4",
            "--printitn", "1",
            "--mu-kappa", "1e-2",
            "--mu-kappatol", "1e-10",
            "--mu-eps", "1e-10",
            "--row-eps-zero-guard", "1e-10",
            "--row-log-zero-guard", "1e-16",
            "--row-eps-active-set", "1e-3",
            "--row-max-backtrack-steps", "10",
            "--row-min-step-size", "1e-7",
            "--row-step-reduction-factor", "0.5",
            "--row-suff-decrease-tolerance", "1e-4",
            "--pdnr-mu", "1e-5",
            "--pdnr-increase-factor", "3.5",
            "--pdnr-decrease-factor", "0.2857142857142857",
            "--pdnr-increase-tolerance", "0.25",
            "--pdnr-decrease-tolerance", "0.75",
            "--pdnr-no-scale-hessian",
            "--pdnr-kernel-option", "1",
            "--pdnr-multi-kernel-size", "4096",
            "--pqnr-lbfgs", "3",
            "--log-file", "",
            "--log-level", "1",
            "--scheduling", "static",
            "--precision", "16",
            "--global-ordinal-type", "int32",
            "--local-ordinal-type", "int32",
            "--sparse-value-type", "int32",
            "--kruskal-value-type", "double"
        };
        // Unused
        /* _default_args.push_back("--mode"); */
        /* _default_args.push_back(""); */
        /* _default_args.push_back("--asciiFormat"); */
        /* _default_args.push_back(""); */
    }

    void CommandLineOptionsUnitTest::TearDown(){}

    TEST_F(CommandLineOptionsUnitTest, get_command_line_options_no_args)
    {
        // Default value
        ASSERT_NO_THROW(_clo.get_command_line_options(_empty_args));
    }

    TEST_F(CommandLineOptionsUnitTest, get_command_line_options_all_args_valid)
    {
        // Default value
        ASSERT_NO_THROW(_clo.get_command_line_options(_default_args));
    }

    TEST_F(CommandLineOptionsUnitTest, check_and_print_unused_args)
    {
        // Default value
        ASSERT_FALSE(sparten::check_and_print_unused_args(_empty_args, std::cout));
        // Try an some argument arg without a value
        _test_args.push_back("--arg");
        ASSERT_TRUE(sparten::check_and_print_unused_args(_test_args, std::cout));
        // Add an argument
        _test_args.push_back("val");
        ASSERT_TRUE(sparten::check_and_print_unused_args(_test_args, std::cout));
    }

    TEST_F(CommandLineOptionsUnitTest, parse_string)
    {
        // Default value
        EXPECT_EQ(sparten::parse_string(_empty_args,"--method","Multiplicative-Update"), "Multiplicative-Update");
        // Non-default value
        _test_args = {"--method", "Quasi-Newton"};
        EXPECT_EQ(sparten::parse_string(_test_args,"--method","Damped-Newton"),"Quasi-Newton");
    }

    TEST_F(CommandLineOptionsUnitTest, parse_uint64_t)
    {
        // Default value
        EXPECT_EQ(sparten::parse_uint64_t(_empty_args,"--maxiters",val_uint64_t,min_uint64_t,max_uint64_t), 20);
        // Value is a negative
        _test_args = {"--maxiters", "-20"};
        ASSERT_THROW(sparten::parse_uint64_t(_test_args,"--maxiters",val_uint64_t,min_uint64_t,max_uint64_t),std::runtime_error);
        // Value is a float -- will get converted.
        _test_args = {"--maxiters", "20.1"};
        ASSERT_EQ(sparten::parse_uint64_t(_test_args,"--maxiters",val_uint64_t,min_uint64_t,max_uint64_t),20);
        // Value is above max
        _test_args = {"--maxiters", "1001"};
        ASSERT_THROW(sparten::parse_uint64_t(_test_args,"--maxiters",val_uint64_t,min_uint64_t,max_uint64_t),std::runtime_error);
        // Value is below min
        _test_args = {"--maxiters", "0"};
        ASSERT_THROW(sparten::parse_uint64_t(_test_args,"--maxiters",val_uint64_t,min_uint64_t,max_uint64_t),std::runtime_error);
    }

    TEST_F(CommandLineOptionsUnitTest, parse_double)
    {
        // Default value
        EXPECT_EQ(sparten::parse_double(_empty_args,"--tol",val_double,min_double,max_double), 1e-10);
        // Value is a string
        _test_args = {"--tol","val"};
        ASSERT_THROW(sparten::parse_double(_test_args,"--tol",val_double,min_double,max_double),std::runtime_error);
        // Value is above max
        _test_args = {"--tol","1.1"};
        ASSERT_THROW(sparten::parse_double(_test_args,"--tol",val_double,min_double,max_double),std::runtime_error);
        // Value is below min
        _test_args = {"--tol","-0.1"};
        ASSERT_THROW(sparten::parse_double(_test_args,"--tol",val_double,min_double,max_double),std::runtime_error);
    }

    TEST_F(CommandLineOptionsUnitTest, parse_bool)
    {
        // Default value
        ASSERT_FALSE(sparten::parse_bool(_empty_args,"--pdnr-no-scale-hessian","--pdnr-scale-hessian",false));
        // Truth table
        _test_args = {"--pdnr-scale-hessian"};
        ASSERT_FALSE(sparten::parse_bool(_test_args,"--pdnr-no-scale-hessian","--pdnr-scale-hessian",true));
        _test_args = {"--pdnr-scale-hessian"};
        ASSERT_FALSE(sparten::parse_bool(_test_args,"--pdnr-no-scale-hessian","--pdnr-scale-hessian",false));
        _test_args = {"--pdnr-no-scale-hessian"};
        ASSERT_TRUE(sparten::parse_bool(_test_args,"--pdnr-no-scale-hessian","--pdnr-scale-hessian",false));
        _test_args = {"--pdnr-no-scale-hessian"};
        ASSERT_TRUE(sparten::parse_bool(_test_args,"--pdnr-no-scale-hessian","--pdnr-scale-hessian",true));
    }

} // end namespace unitTest
