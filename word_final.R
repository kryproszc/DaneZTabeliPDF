library(readxl)
library(officer)
library(dplyr)
library(e1071)
library(moments)
library(ggplot2)

generuj_word = function(folder,wartosci_path,sciezka_zapisu){
  wartosci_bazowe = read_excel(wartosci_path)
  
  doc <- read_docx()
  
  pliki_csv <- list.files(path = sciezka, pattern = "\\.csv$", full.names = TRUE)
  
  lista_dataframe <- list()
  
  for (plik in pliki_csv) {
    dataframe <- read.csv(plik, header = TRUE,sep=";")  # Wczytaj plik CSV
    lista_dataframe[[basename(plik)]] <- dataframe  # Dodaj dataframe do listy
  }
  
  n = 1
  
  z9_1 = c()
  z9_2 = c()
  z9_3 = c()
  z9_4 = c()
  
  for(df in lista_dataframe){
    sr1 = mean(df$Zysk.brutto,na.rm=T)
    sr2 = mean(df$Zysk.netto,na.rm=T)
    sr3 = mean(df$Zysk.podatek,na.rm=T)
    sr4 = mean(df$Zysk.podatek.nowy,na.rm=T)
    
    sd1 = sd(df$Zysk.brutto,na.rm=T)
    sd2 = sd(df$Zysk.netto,na.rm=T)
    sd3 = sd(df$Zysk.podatek,na.rm=T)
    sd4 = sd(df$Zysk.podatek.nowy,na.rm=T)
    
    as1 = skewness(df$Zysk.brutto,na.rm = T)
    as2 = skewness(df$Zysk.netto,na.rm = T)
    as3 = skewness(df$Zysk.podatek,na.rm = T)
    as4 = skewness(df$Zysk.podatek.nowy,na.rm = T)
    
    k1 = kurtosis(df$Zysk.brutto,na.rm = T)
    k2 = kurtosis(df$Zysk.netto,na.rm = T)
    k3 = kurtosis(df$Zysk.podatek,na.rm = T)
    k4 = kurtosis(df$Zysk.podatek.nowy,na.rm = T)
    
    min1 = min(df$Zysk.brutto,na.rm=T)
    min2 = min(df$Zysk.netto,na.rm=T)
    min3 = min(df$Zysk.podatek,na.rm=T)
    min4 = min(df$Zysk.podatek.nowy,na.rm=T)
    
    p1_001 = quantile(df$Zysk.brutto,na.rm = T,0.01)
    p2_001 = quantile(df$Zysk.netto,na.rm = T,0.01)
    p3_001 = quantile(df$Zysk.podatek,na.rm = T,0.01)
    p4_001 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.01)
    
    p1_005 = quantile(df$Zysk.brutto,na.rm = T,0.05)
    p2_005 = quantile(df$Zysk.netto,na.rm = T,0.05)
    p3_005 = quantile(df$Zysk.podatek,na.rm = T,0.05)
    p4_005 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.05)
    
    p1_01 = quantile(df$Zysk.brutto,na.rm = T,0.1)
    p2_01 = quantile(df$Zysk.netto,na.rm = T,0.1)
    p3_01 = quantile(df$Zysk.podatek,na.rm = T,0.1)
    p4_01 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.1)
    
    p1_02 = quantile(df$Zysk.brutto,na.rm = T,0.2)
    p2_02 = quantile(df$Zysk.netto,na.rm = T,0.2)
    p3_02 = quantile(df$Zysk.podatek,na.rm = T,0.2)
    p4_02 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.2)
    
    p1_03 = quantile(df$Zysk.brutto,na.rm = T,0.3)
    p2_03 = quantile(df$Zysk.netto,na.rm = T,0.3)
    p3_03 = quantile(df$Zysk.podatek,na.rm = T,0.3)
    p4_03 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.3)
    
    p1_04 = quantile(df$Zysk.brutto,na.rm = T,0.4)
    p2_04 = quantile(df$Zysk.netto,na.rm = T,0.4)
    p3_04 = quantile(df$Zysk.podatek,na.rm = T,0.4)
    p4_04 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.4)
    
    p1_05 = quantile(df$Zysk.brutto,na.rm = T,0.5)
    p2_05 = quantile(df$Zysk.netto,na.rm = T,0.5)
    p3_05 = quantile(df$Zysk.podatek,na.rm = T,0.5)
    p4_05 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.5)
    
    p1_06 = quantile(df$Zysk.brutto,na.rm = T,0.6)
    p2_06 = quantile(df$Zysk.netto,na.rm = T,0.6)
    p3_06 = quantile(df$Zysk.podatek,na.rm = T,0.6)
    p4_06 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.6)
    
    p1_07 = quantile(df$Zysk.brutto,na.rm = T,0.7)
    p2_07 = quantile(df$Zysk.netto,na.rm = T,0.7)
    p3_07 = quantile(df$Zysk.podatek,na.rm = T,0.7)
    p4_07 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.7)
    
    p1_08 = quantile(df$Zysk.brutto,na.rm = T,0.8)
    p2_08 = quantile(df$Zysk.netto,na.rm = T,0.8)
    p3_08 = quantile(df$Zysk.podatek,na.rm = T,0.8)
    p4_08 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.8)
    
    p1_09 = quantile(df$Zysk.brutto,na.rm = T,0.9)
    p2_09 = quantile(df$Zysk.netto,na.rm = T,0.9)
    p3_09 = quantile(df$Zysk.podatek,na.rm = T,0.9)
    p4_09 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.9)
    
    p1_095 = quantile(df$Zysk.brutto,na.rm = T,0.95)
    p2_095 = quantile(df$Zysk.netto,na.rm = T,0.95)
    p3_095 = quantile(df$Zysk.podatek,na.rm = T,0.95)
    p4_095 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.95)
    
    p1_0995 = quantile(df$Zysk.brutto,na.rm = T,0.995)
    p2_0995 = quantile(df$Zysk.netto,na.rm = T,0.995)
    p3_0995 = quantile(df$Zysk.podatek,na.rm = T,0.995)
    p4_0995 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.995)
    
    p1_0999 = quantile(df$Zysk.brutto,na.rm = T,0.999)
    p2_0999 = quantile(df$Zysk.netto,na.rm = T,0.999)
    p3_0999 = quantile(df$Zysk.podatek,na.rm = T,0.999)
    p4_0999 = quantile(df$Zysk.podatek.nowy,na.rm = T,0.999)
    
    
    max1 = max(df$Zysk.brutto,na.rm=T)
    max2 = max(df$Zysk.netto,na.rm=T)
    max3 = max(df$Zysk.podatek,na.rm=T)
    max4 = max(df$Zysk.podatek.nowy,na.rm=T)
    
    dane = data.frame(stat = c("Średnia","Odchylenie","Kurtoza","Skośność","Min","0.01","0.05","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","0.95","0.995","0.999","Max"),
                      z.brutto = c(sr1,sd1,k1,as1,min1,p1_001,p1_005,p1_01,p1_02,p1_03,p1_04,p1_05,p1_06,p1_07,p1_08,p1_09,p1_095,p1_0995,p1_0999,max1), 
                      z.netto = c(sr2,sd2,k2,as2,min2,p2_001,p2_005,p2_01,p2_02,p2_03,p2_04,p2_05,p2_06,p2_07,p2_08,p2_09,p2_095,p2_0995,p2_0999,max2), 
                      z.podatek = c(sr3,sd3,k3,as3,min3,p3_001,p3_005,p3_01,p3_02,p3_03,p3_04,p3_05,p3_06,p3_07,p3_08,p3_09,p3_095,p3_0995,p3_0999,max3),
                      z.podatek.nowy = c(sr4,sd4,k4,as4,min4,p4_001,p4_005,p4_01,p4_02,p4_03,p4_04,p4_05,p4_06,p4_07,p4_08,p4_09,p4_095,p4_0995,p4_0999,max4))
    
    colnames(dane) = c("Statystyka","Zysk brutto","Zysk netto", "Zysk podatek", "Zysk podatek nowy")
    
    doc = doc %>% body_add_fpar(fpar(paste("Wyniki spolki",wartosci_bazowe[n,1])))
    
    doc <- doc %>% body_add_table(dane,style = "table_template")
    
    h1 = ggplot(data.frame(x = df$Zysk.brutto), aes(x)) +
      geom_histogram(fill = "skyblue", color = "black", bins = 20) +
      labs(title = "Zysk brutto", x = "Wartość", y = "Częstość")
    
    doc = doc %>% body_add_plot(value = print(h1, device = "off"), width = 5, height = 4)
    
    h2 = ggplot(data.frame(x = df$Zysk.netto), aes(x)) +
      geom_histogram(fill = "skyblue", color = "black", bins = 20) +
      labs(title = "Zysk netto", x = "Wartość", y = "Częstość")
    
    doc = doc %>% body_add_plot(value = print(h2, device = "off"), width = 5, height = 4)
    
    h3 = ggplot(data.frame(x = df$Zysk.podatek), aes(x)) +
      geom_histogram(fill = "skyblue", color = "black", bins = 20) +
      labs(title = "Zysk podatek", x = "Wartość", y = "Częstość")
    
    doc = doc %>% body_add_plot(value = print(h3, device = "off"), width = 5, height = 4)
    
    h4 = ggplot(data.frame(x = df$Zysk.netto), aes(x)) +
      geom_histogram(fill = "skyblue", color = "black", bins = 20) +
      labs(title = "Zysk podatek nowy", x = "Wartość", y = "Częstość")
    
    doc = doc %>% body_add_plot(value = print(h4, device = "off"), width = 5, height = 4)
    
    x = c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.995,0.999)
    y1 = c(p1_001,p1_005,p1_01,p1_02,p1_03,p1_04,p1_05,p1_06,p1_07,p1_08,p1_09,p1_095,p1_0995,p1_0999)
    y2 = c(p2_001,p2_005,p2_01,p2_02,p2_03,p2_04,p2_05,p2_06,p2_07,p2_08,p2_09,p2_095,p2_0995,p2_0999)
    y3 = c(p3_001,p3_005,p3_01,p3_02,p3_03,p3_04,p3_05,p3_06,p3_07,p3_08,p3_09,p3_095,p3_0995,p3_0999)
    y4 = c(p4_001,p4_005,p4_01,p4_02,p4_03,p4_04,p4_05,p4_06,p4_07,p4_08,p4_09,p4_095,p4_0995,p4_0999)
    
    liniowy = ggplot(data.frame(x = x, y1 = y1, y2 = y2, y3 = y3, y4 = y4), aes(x)) +
      geom_line(aes(y = y1, color = "y1"), linetype = "solid") +
      geom_line(aes(y = y2, color = "y2"), linetype = "dashed") +
      geom_line(aes(y = y3, color = "y3"), linetype = "dotted") +
      geom_line(aes(y = y4, color = "y4"), linetype = "dotdash") +
      labs(title = "Wykres liniowy kwantyli", x = "Wartość x", y = "Wartość y") +
      scale_color_manual(name = "Seria danych",
                         values = c("y1" = "blue", "y2" = "red", "y3" = "green", "y4" = "orange"),
                         labels = c("Zysk brutto", "Zysk netto", "Zysk podatek", "Zysk podatek nowy"))
    
    print(liniowy)
    
    doc = doc %>% body_add_plot(value = print(liniowy, device = "off"), width = 7, height = 5)
    
    z9_1 = c(z9_1,p1_09)
    z9_2 = c(z9_2,p2_09)
    z9_3 = c(z9_3,p3_09)
    z9_4 = c(z9_4,p4_09)
    
    n = n + 1
  }
  
  wartosci_bazowe$`Zysk brutto` = z9_1
  wartosci_bazowe$`Zysk netto` = z9_2
  wartosci_bazowe$`Zysk podatek` = z9_3
  wartosci_bazowe$`Zysk podatek nowy` = z9_4
  
  doc = doc %>% body_add_fpar(fpar("Podsumowanie"))
  
  doc <- doc %>% body_add_table(wartosci_bazowe,style = "table_template")
  
  print(doc, target = paste(sciezka_zapisu,"podsumowanie.docx",sep=""))
}

sciezka = "/Users/Szczesny/Desktop/BootChainladder/Test word"
wp = '/Users/Szczesny/Desktop/BootChainladder/Test word/Wartosci_bazowe.xlsx'
sciezka_zapisu = "/Users/Szczesny/Desktop/BootChainladder/Test word/"
generuj_word(sciezka,wp,sciezka_zapisu)

