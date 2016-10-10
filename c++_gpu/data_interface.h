// data_interface.h
// function:

#ifndef DATA_INTERFACE_H
#define DATA_INTERFACE_H



#include <vector>



using namespace std;




// data and model:
void data_load_simu();
void data_load_real();
void model_save();


// error:
void error_init();
void error_save();
void error_save_online();




#endif

// end of data_interface.h
