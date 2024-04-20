
exponsure_longitude<-list()
exponsure_latitude<-list()
exponsure_sum_value<-list()
exponsure_insurance<-list()
exponsure_reassurance<-list()
for(woj in 1:17){
  exponsure_longitude[[woj]]<-list()
  exponsure_latitude[[woj]]<-list()
  exponsure_sum_value[[woj]]<-list()
  exponsure_insurance[[woj]]<-list()
  exponsure_reassurance[[woj]]<-list()
  for(month in 1:12){
    n_num<-runif(1,1000,20000)
    exponsure_latitude[[woj]][[month]]<-runif(n_num,49.0,54.50)
    exponsure_longitude[[woj]][[month]]<-runif(n_num,16.07,24.09)
    exponsure_sum_value[[woj]][[month]]<-runif(n_num,2000,500000)
    exponsure_insurance[[woj]][[month]]<-round(runif(n_num,0,3),0)
    exponsure_reassurance[[woj]][[month]]<-round(runif(n_num,0,3),0)
    
  }
}

len<-c()
  i = 0
for(woj in 1:17){
  for(month in 1:12){
    i=i+1
    len[i]<-length(exponsure_latitude[[woj]][[month]])
  }}

#prawdopodobienstwa wybuchow pozarow
list_list_wyb<-c()
  for(woj in 1:17){
    list_list_wyb[[woj]]<-runif(12,5.44e-05,9.44e-05)
  }
  
#wielkosc pozaru
  wielkosc_pozaru<-c()
    for(i in 1:2){
      wielkosc_pozaru[[i]]<-runif(5000,0,1)
    }
    
#pawdopoodbienstwo rozprzestrzenienia
    fire_spread_prob_vec<-list()
      
      fire_spread_prob_vec[[1]]<-runif(9,0.009,0.2)
      fire_spread_prob_vec[[2]]<-runif(9,0.01,0.2)
      fire_spread_prob_vec[[3]]<-runif(9,0.01,0.15)
      fire_spread_prob_vec[[4]]<-runif(9,0.08,0.4)
      
      conditional_mean_trend_parameters<-c(0.14,0.44)
      conditional_Cov<-c(1.25,2.29)
      
      
#reasekuracja
      fakultatywna_input_num<-c()
        fakultatywna_input_num[[1]]<-c(0,1,2,3)
        fakultatywna_input_num[[2]]<-c(0,1,2,3)
        fakultatywna_input_num[[3]]<-c(0,1,2,3)
        fakultatywna_input_num[[4]]<-c(0,1,2,3)
        
        fakultatywna_input_val<-list()
        for( i in 1:4){
          fakultatywna_input_val[[i]]<-list()
          for(j in 1:4){
            fakultatywna_input_val[[i]][[j]]<-c(0.2,10000000)
            fakultatywna_input_val[[i]][[j]]<-c(0.4,10000000)
            fakultatywna_input_val[[i]][[j]]<-c(0.2,10000000)
            fakultatywna_input_val[[i]][[j]]<-c(0.8,10000000)
          }
        }
        
        obligatoryjna_input_risk<-list()
          obligatoryjna_input_risk[[1]]<-c(2500000,10000000,1,1)
          obligatoryjna_input_risk[[2]]<-c(2500000,10000000,1,1)
          obligatoryjna_input_risk[[3]]<-c(2500000,10000000,1,1)
          obligatoryjna_input_risk[[4]]<-c(2500000,10000000,1,1)
          
          obligatoryjna_input_event<-list()
          obligatoryjna_input_event[[1]]<-c(25000000,100000000,1,1)
          obligatoryjna_input_event[[2]]<-c(25000000,100000000,1,1)
          obligatoryjna_input_event[[3]]<-c(25000000,100000000,1,1)
          obligatoryjna_input_event[[4]]<-c(25000000,100000000,1,1)
          obligatoryjna_input_event