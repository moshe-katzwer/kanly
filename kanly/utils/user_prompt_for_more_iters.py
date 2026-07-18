from __future__ import absolute_import, print_function

import numpy as np


def user_prompt_for_more_iters_method(message, do_prompt, print_not_converged=True,
                                      assert_even=False, accept_list=False):
    """Prompt the user for more iterations before exiting"""

    try:
        if do_prompt:
            print((f"\nAlgorithm has not converged." if print_not_converged else '\n') + message)
            while True:
                more_iters = input(
                    f"\nInput the number of additional iterations (or nothing to terminate):\n")
                if more_iters == '':
                    return 0
                else:
                    try:
                        if accept_list:
                            more_iters = more_iters.split(',')
                            incremental_iters = [int(x) for x in more_iters]
                            assert np.all(np.array(incremental_iters) > 0)
                            if assert_even:
                                if np.any(np.array(incremental_iters) % 2):
                                    raise Exception
                            return incremental_iters
                        else:
                            incremental_iters = int(more_iters)
                            assert incremental_iters > 0
                            if assert_even and incremental_iters % 2:
                                raise Exception
                            return incremental_iters
                    except:
                        if accept_list:
                            print(f'f{str(more_iters)} was not a valid input; please input positive"'
                                  f'f" {"even " if assert_even else ""}integers')
                        else:
                            if incremental_iters < 0:
                                print(f"{more_iters} is an invalid input, input a positive integer (or nothing to terminate)")
                            elif assert_even and incremental_iters % 2:
                                print(f"{more_iters} is not even, input an even positive integer (or nothing to terminate)")
        else:
            if accept_list:
                return []
            else:
                return 0

    except Exception as e:
        print(f"\nException{str(e)}, trying again...\n")
        return user_prompt_for_more_iters_method(
            message, do_prompt, print_not_converged=print_not_converged,
            assert_even=assert_even, accept_list=accept_list)