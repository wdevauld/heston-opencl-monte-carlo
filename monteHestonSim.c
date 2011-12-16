#include <fcntl.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

// Size used for character buffer to read device information
#define SCRATCH_SIZE 2048

//Constants for what pricing we want
#define PRICE 0
#define CALL 1
#define PUT 2
#define USAGE "monteHestonSim OPTIONS\n \
\t-g [NUM]  - Set the number of working groups, default: device maximum \n\
\t-c        - Use CPU instead of GPU \n\
\t-h        - Display this message \n\
\t-v        - Display extra information\n \
\t-i [NUM]  - Set initial price, default: 10\n \
\t-r [NUM]  - Asset rate of return/drift, default: 0.05\n \
\t-m [NUM]  - Volatility mean reversion level, default: 0.2 \n \
\t-l [NUM]  - Mean reversion rate, default 1.2\n \
\t-s [NUM]  - Volatility of volatility, default 0.1\n \
\t-k [NUM]  - Strike, used only if -C or -P are used, default: 10\n \
\t-d [NUM]  - Number of increments to use along the path, default: 500\n \
\t-p [NUM]  - log2 of paths to generate, default: 10 (2^10 = 1024 paths)\n \
\t-P -C     - Caluclate Put or Call payoff\n"

int main(int argc, char** argv)
{
    char scratch[SCRATCH_SIZE];    // Used to pull string from OpenCL 
    int err;               // error code returned from api calls
    int opt, flags;        // Used for option parsing
    int user_work_groups = 0;   // User selected work groups
    int log_2_size = 10;
    //TODO CAREFUL, we aren't checking if 2^log_2_size > INT_MAX!
    // so keep log_2_size under 28ish.
    int data_size;               //Number of paths to simulate
    bool cpu = false;    
    bool verbose = false;
    int payoff = PRICE;
      
    int seed; // Outgoing seed data for the random number generator 
    float *results;   // results returned from computation (prices) 
    float band_width; //The size of one half of the 95% confidence interval

    size_t work_count;     // need a variable to pass in the size of work 
    size_t work_groups;    // number of work work_group
    size_t fileSize;       // Size of the CL program to be read in

    //OpenCL handles to various objects
    cl_device_id devices[10];    
    cl_context context;          
    cl_command_queue commands;  
    cl_program program;        
    cl_kernel seeds_kernel;  //Generates an array of uniforms from a single seed 
    cl_kernel paths_kernel;  //Price path realization
    cl_kernel payoff_kernel; //Applies the appropriate payoff function
    cl_kernel mean_stddev_kernel;  //Calculates the mean and standard deviation

    cl_mem seed_output;  //An array of uniform random variables over [0, INT_MAX]
    cl_mem price_output; //An array of price realizations from the Heston Model
    cl_mem mean_stddev; //A 2-element array, first is mean, second is standard dev

    int i = 0;

    //Parameters for price path simulation
    float initial_price = 10;
    float r = 0.05;
    float mu = 0.2;
    float lambda = 1.2;
    float sigma = 0.1;
    int divisions = 500;
    //Option parameters
    float strike = 10;

    //These clock milestones represent the clock cycles completed
    // when the particular event is completed
    clock_t start, c_device, c_context, c_command, c_read,
            c_program, c_build, c_kernel, c_copydata, c_execute,
            c_readback;    

    while ((opt = getopt(argc, argv, "d:hg:i:cvr:m:l:s:k:p:PC")) != -1)
    {
        switch(opt) 
        {
            case 'g':
                user_work_groups = atoi(optarg);
                break;
            case 'i':
                initial_price = atof(optarg); 
                break;
            case 'c':
                cpu = true;
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
                printf(USAGE);
                exit(1);
            case 'r':
                r = atof(optarg);
                break;
            case 'm':
                mu = atof(optarg);
                break;
            case 'l':
                lambda = atof(optarg);
                break;
            case 's':
                sigma = atof(optarg);
                break;
            case 'k':
                strike = atof(optarg);
                break;
            case 'p':
                log_2_size = atoi(optarg);
                break;
            case 'd':
                divisions = atoi(optarg);
                break;
            case 'P':
                payoff = PUT;
                break;
            case 'C':
                payoff = CALL;
                break;

        }
    }
    //The results (mean and stddev)    
    results = malloc(sizeof(float) * 2);
    
    data_size = pow(2, log_2_size);  //Amount of price paths to calculate
    work_count = data_size;
  
    srand((unsigned)time(0));
    seed = rand(); 

    if(verbose) {
        printf("Using initial price: %f\n", initial_price);  
        printf("Using drift: %f\n", r);  
        printf("Using mean reversion level: %f\n", mu);  
        printf("Using mean reversion rate: %f\n", lambda);  
        printf("Simulating %d paths with %d increments each\n", data_size, divisions);
        switch(payoff) 
        {
            case CALL:
                printf("Calculating Call Option payoffs with strike: %f\n", strike);
                break;
            case PUT:
                printf("Calculating Put Option payoffs with strike: %f\n", strike);
                break;
            case PRICE:
                printf("Calculating ending prices\n");
                break;
        }
    }

    start = clock();

    // Attempt to get a GPU computing device 
    //If you don't have a GPU, use CL_DEVICE_TYPE_CPU
    err = clGetDeviceIDs(NULL, (cpu ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU),
                         1, devices, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Could not find a GPU device\n");
        return EXIT_FAILURE;
    }
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, SCRATCH_SIZE, scratch,NULL); 

    if(verbose) printf("Using device: %s\n", scratch); 

    c_device = clock();
    
    // Create a context 
    context = clCreateContext(0, 1, devices, NULL, NULL, &err);
    if (!context)
    {
        printf("Unable to create context\n");
        return EXIT_FAILURE;
    }

    c_context = clock();

    // Create a command queue for the context 
    commands = clCreateCommandQueue(context, devices[0], 0, &err);
    if (!commands)
    {
        printf("Could not create a command queue\n");
        return EXIT_FAILURE;
    }

    c_command = clock();

    //Read in a CL program from the ocl file
    FILE* oclSource = fopen("heston_realizations.ocl","rb");
    if(!oclSource)
    {
        printf("Could not open \"heston_realizations.ocl\" file\n");
        return EXIT_FAILURE;
    }

    //Determine the length of the file
    fseek(oclSource, 0, SEEK_END);
    fileSize = ftell(oclSource);
    fseek(oclSource, 0, SEEK_SET);
    
    char* inMemorySource = (char *)malloc(fileSize + 1);
    fread(inMemorySource, fileSize, 1, oclSource);

    c_read = clock();

    program = clCreateProgramWithSource(context, 1, (const char **) & inMemorySource, NULL, &err);
    if (!program)
    {
        printf("Unable to create OpenCL program\n");
        return EXIT_FAILURE;
    }

    c_program = clock();

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        printf("Could not build program %d\n", err);
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(scratch), scratch, &len);
        printf("%s\n", scratch);
        exit(1);
    }

    c_build = clock();

    //build a device specific kernels
    seeds_kernel = clCreateKernel(program, "uniformSeeds", &err);
    if (!seeds_kernel || err != CL_SUCCESS)
    {
        printf("Failed to create uniform seeding kernel %d\n", err);
        exit(1);
    }
 
    paths_kernel = clCreateKernel(program, "hestonSimulation", &err);
    if (!paths_kernel || err != CL_SUCCESS)
    {
        printf("Failed to create path simulation kernel %d\n", err);
        exit(1);
    }

    mean_stddev_kernel = clCreateKernel(program, "meanAndStandardDeviation", &err);
    if (!paths_kernel || err != CL_SUCCESS)
    {
        printf("Failed to create mean and standard deviation kernel %d\n", err);
        exit(1);
    }

    //We will swap out one of the kernels in the chain depending on user input
    switch(payoff)
    {
        case PRICE:
            payoff_kernel = clCreateKernel(program, "straightPrice", &err); 
            break;
        case CALL:
            payoff_kernel = clCreateKernel(program, "vanillaCall", &err); 
            break;
        case PUT:
            payoff_kernel = clCreateKernel(program, "vanillaPut", &err); 
            break;
        default:
            printf("Error determining appropriate pricing function");  
            exit(1);
    } 

    if (!payoff_kernel || err != CL_SUCCESS)
    {
        printf("Failed to create payoff kernel %d\n", err);
        exit(1);
    }

    c_kernel = clock();
    // Create the input and output arrays in device memory for our calculation
    seed_output  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(int) * data_size, NULL, NULL);
    price_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, NULL);
    mean_stddev = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 2, NULL, NULL);

    if (!seed_output|| !price_output || ! mean_stddev)
    {
        printf("Failed to allocate device memory\n");
        exit(1);
    }    
    
    err = 0;
    err = clSetKernelArg(seeds_kernel, 0, sizeof(int), &seed);
    err = clSetKernelArg(seeds_kernel, 1, sizeof(int), &data_size);
    err |= clSetKernelArg(seeds_kernel, 2, sizeof(cl_mem), &seed_output);
    if(err != CL_SUCCESS) {
        printf("Could not set the arguments to the seeding kernel %d\n", err);
        exit(1);
    }

    err |= clSetKernelArg(paths_kernel, 0, sizeof(cl_mem), &seed_output);
    err |= clSetKernelArg(paths_kernel, 1, sizeof(cl_mem), &price_output);
    err |= clSetKernelArg(paths_kernel, 2, sizeof(float), &initial_price);
    err |= clSetKernelArg(paths_kernel, 3, sizeof(float), &r);
    err |= clSetKernelArg(paths_kernel, 4, sizeof(float), &mu);
    err |= clSetKernelArg(paths_kernel, 5, sizeof(float), &lambda);
    err |= clSetKernelArg(paths_kernel, 6, sizeof(float), &sigma);
    err |= clSetKernelArg(paths_kernel, 7, sizeof(int), &divisions);
    if (err != CL_SUCCESS)
    {
        printf("Could not set the arguments to the path kernel%d\n", err);
        exit(1);
    }

    err |= clSetKernelArg(payoff_kernel, 0, sizeof(cl_mem), &price_output);  
    err |= clSetKernelArg(payoff_kernel, 1, sizeof(float), &strike);  
    if (err != CL_SUCCESS)
    {
        printf("Could not set the arguments to the payoff kernel%d\n", err);
        exit(1);
    }

    err |= clSetKernelArg(mean_stddev_kernel, 0, sizeof(int), &data_size);  
    err |= clSetKernelArg(mean_stddev_kernel, 1, sizeof(cl_mem), &price_output);  
    err |= clSetKernelArg(mean_stddev_kernel, 2, sizeof(cl_mem), &mean_stddev);  
    
    if (err != CL_SUCCESS)
    {
        printf("Could not set the arguments to the Mean and Standard Deviation kernel %d\n", err);
        exit(1);
    }
    c_copydata = clock();

    //Determine maximum working groups
    err = clGetKernelWorkGroupInfo(paths_kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(work_groups), &work_groups, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Could not determine maximum work groups because: %d\n", err);
        exit(1);
    }
    if(verbose) printf("Maximum for device: %i\n", (int)work_groups);

    //Override the value for SCIENCE!
    if(user_work_groups > 0 && user_work_groups < work_groups) {
        work_groups = user_work_groups;
    }

    if (verbose) printf("Using: %i work groups\n", (int)work_groups);

    //Enqueue a task to generate all the seeds
    err = clEnqueueTask(commands, seeds_kernel, 0, NULL, NULL);
    err |= clEnqueueBarrier(commands); 
    
    //Enqueue the generation of price paths
    err |= clEnqueueNDRangeKernel(commands, paths_kernel, 1, NULL, &work_count, &work_groups, 0, NULL, NULL);
    err |= clEnqueueBarrier(commands); 
    
    err |= clEnqueueNDRangeKernel(commands, payoff_kernel, 1, NULL, &work_count, &work_groups, 0, NULL, NULL);
    err |= clEnqueueBarrier(commands); 
   
    err |= clEnqueueTask(commands, mean_stddev_kernel, 0, NULL, NULL);
    err |= clFinish(commands);

    c_execute = clock();

    /* If you need to get the payoffs pushed to stdout, uncomment this
       TODO make this an option 
    err = clEnqueueReadBuffer(commands, price_output, CL_TRUE, 0, sizeof(float) * data_size, results, 0, NULL, NULL);
    for(i = 0; i<data_size; i++) printf("%f\n", results[i]);
    */
    
    // Read the data back into local memory 
    err = clEnqueueReadBuffer(commands, mean_stddev, CL_TRUE, 0, sizeof(float) * 2, results, 0, NULL, NULL);

    c_readback = clock();
    
    if (err != CL_SUCCESS)
    {
        printf("Failed to copy output:  %d\n", err);
        exit(1);
    }
    if(verbose) printf("Payoff Standard Deviation: %f\n", results[1]); 
    band_width = 1.96 * results[1] / sqrt(data_size);
    printf("Expected Payoff: %f\n95%% Confidence band: [%f,%f]\n", results[0], results[0] - band_width, results[0] + band_width);
     
    //print out the results
    if(verbose) {
        //We care more about timing data
        printf("%8.5f\tseconds to find devices\n", (double)(c_device - start)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to create context\n", (double)(c_context - c_device)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to create command queue\n", (double)(c_command - c_context)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to read in program\n", (double)(c_read - c_command)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to find create program\n", (double)(c_program - c_read)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to build program\n", (double)(c_build- c_program)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to construct kernel\n", (double)(c_kernel - c_build)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to populate input queues\n", (double)(c_copydata - c_kernel)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to execute program\n", (double)(c_execute - c_copydata)/CLOCKS_PER_SEC);
        printf("%8.5f\tseconds to read output queues\n", (double)(c_readback - c_execute)/CLOCKS_PER_SEC);
    }
    // Shutdown and cleanup
    //
    clReleaseMemObject(seed_output);
//    clReleaseMemObject(output);
    clReleaseProgram(program);
//    clReleaseKernel(paths_kernel);
    clReleaseKernel(seeds_kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

