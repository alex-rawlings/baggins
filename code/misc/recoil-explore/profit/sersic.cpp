#include "profit/profit.h"

int main(){
    profit::init();
    profit::Model model(10.0, 10.0);

    profit::ProfilePtr sersic_profile = model.add_profile("sersic");

    sersic_profile.parameter("xcen", 0.0);
    sersic_profile.parameter("ycen", 0.0);
    sersic_profile.parameter("axrt", 0.5);
    sersic_profile.parameter("nser", 4.0);

    profit::Image result = model.evaluate();
    profit::finish();
}