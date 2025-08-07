
import cProfile
import pstats
import io


def profile_function(func, subfunction_to_profile, *args, **kwargs, ):
    """
    Profiles a given function and returns the time spent in that function.

    Parameters:
    - func: The function to run.
    - subfunction_to_profile: the subfunction to extract stats for.
    - *args: Positional arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.
    

    Returns:
    - A dictionary containing the total time spent in the function and the number of calls.
    """
    # Create a profiler
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    # Call the function with the provided arguments
    func(*args, **kwargs)

    profiler.disable()  # Stop profiling

    # Create a stream to hold the profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()  # Print the profiling results

    # Parse the output to find the specific function's stats
    function_name = subfunction_to_profile.__name__
    function_stats = {}

    for line in s.getvalue().splitlines():
        #print(line)
        if function_name in line:
            parts = line.split()
            # Extract the relevant statistics
            function_stats['ncalls'] = int(parts[0])  # Number of calls
            function_stats['tottime'] = float(parts[2])  # Total time spent in the function
            function_stats['percall'] = float(parts[2]) / int(parts[0]) if int(parts[0]) > 0 else 0  # Time per call
            function_stats['cumtime'] = float(parts[3])  # Cumulative time spent in the function
            break

    return function_stats
