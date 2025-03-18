#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <gmp.h>
#include <omp.h>
#include <unistd.h>

// Check if p is safe prime: if p and (p-1)/2 are prime
int is_safe_prime(mpz_t p)
{
    if (mpz_probab_prime_p(p, 15) == 0)
        return 0;
    
    mpz_t q;
    mpz_init(q);
    
    // Calculation q = (p-1)/2
    mpz_sub_ui(q, p, 1);
    mpz_divexact_ui(q, q, 2);
    
    // Check if q is prime
    int result = mpz_probab_prime_p(q, 15);
    
    mpz_clear(q);
    return result;
}

int main()
{
    clock_t start_time = clock();
    int flag = 0;
    
    // Use the max available number of threads
    #pragma omp parallel shared(flag, start_time)
    {
        mpz_t p;
        mpz_init(p);
        
        // Creating a random number generator with a unique seed for each thread
        gmp_randstate_t local_state;
        gmp_randinit_mt(local_state);  // Using Mersenne Twister for better quality random numbers
        
        // Unique seed for each thread based on time, thread id and process id
        unsigned long seed = time(NULL) * (omp_get_thread_num() + 1) * getpid();
        gmp_randseed_ui(local_state, seed);
        
        while (!flag)
        {
            // Search directly for 2*q+1 where q is prime
            mpz_urandomb(p, local_state, 2047);
            mpz_nextprime(p, p);  // Find a prime q
            
            // Calculate p = 2q + 1
            mpz_mul_ui(p, p, 2);
            mpz_add_ui(p, p, 1);
            
            // Use the special safe prime control function
            if (is_safe_prime(p))
            {
                #pragma omp critical
                {
                    if (!flag)
                    {
                        // Calculating time accurately
                        double end_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
                        
                        gmp_printf("%Zd Safe Prime\n", p);
                        printf("\n%lf sec\n", end_time);
                        flag = 1;
                    }
                }
            }
            
            // Added short pause for better parallelism performance on multiple threads
            if (omp_get_thread_num() % 4 == 0)
                usleep(1);
        }
        
        mpz_clear(p);
        gmp_randclear(local_state);
    }
    
    return 0;
}
