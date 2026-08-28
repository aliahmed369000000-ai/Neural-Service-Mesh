# NSM Security Simulation Target
# This file contains intentional patterns for security audit testing.

def vulnerable_function(user_input):
    # Intentional Command Injection vulnerability for testing
    import os
    os.system("ping " + user_input)

def leaked_secret():
    # Intentional secret leak pattern for testing
    api_key = "ghp_TestToken12345678901234567890123456"
    return api_key

if __name__ == "__main__":
    vulnerable_function("127.0.0.1")
