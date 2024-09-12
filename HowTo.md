# Global directories

It's common to use global environment variables that can be used by programs. 

Environment variables helps your computer know where to find certain files or where to put them without you needing to specify the path every time. It makes your work easier and more organized.

If you are using a bash environment these can be specified in the ```.bashrc``` file (rc = run commands). This file runs every time you open a new terminal window and sets up your environment.

This is usually stored at ```~/.bashrc```  

## Example

### Adding the directories
We often want to specify a place to store data and outputs. Let's do this as an example.

Edit the file with your preffered editor e.g.

```bash
nano ~/.bashrc
```

Add the following lines to the end of the file:
```bash
export DATA_DIR=/home/user/mydata
export OUT_DIR=/home/user/myoutputs
```

Save the file and exit.
You can then apply these files by either opening a new terminal or enerting ```source ~/.bashrc```. 

#### Testing and usage

Try typing the following in your terminal:
```bash
ls $DATA_DIR
```
```bash
echo $DATA_DIR
echo $OUT_DIR
```

You can now easily move things using these too.
```bash
cp myfile.txt $OUT_DIR
```

## Using this in programs

### Python
To access and use these environment variables in python we need to use the `os` modile.
```python
import os

data_dir = os.getenv('DATA_DIR')
out_dir = os.getenv('OUT_DIR')

print(f"Data directory: {data_dir}")
print(f"Output directory: {out_dir}")
```

### C++

In C++, we need to use the `getenv` function from the `cstdlib` library.
```C++
#include <iostream>
#include <cstdlib>

int main() {
    const char* data_dir = std::getenv("DATA_DIR"); //if you want to change the dir don't use const
    const char* out_dir = std::getenv("OUT_DIR");


    //checks and prints the dir or error
    if (data_dir && out_dir) {                  
        std::cout << "Data directory: " << data_dir << std::endl;
        std::cout << "Output directory: " << out_dir << std::endl;
    } else {
        std::cout << "Environment variables not set." << std::endl;
    }

    return 0;
}
```