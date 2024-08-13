library(shiny)
library(leaflet)
library(dplyr)
library(geosphere)
library(ggplot2)
library(DT)
library(shinycssloaders)
library(shinyjs)
library(officer)
library(rmarkdown)
library(shinydashboard)

# Funkcje do obliczeń i generowania raportów
calculate_statistics <- function(data) {
  quantiles <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995)
  stats <- data.frame(
    Statystyki = c("min", "max", quantiles, "mean", "0.995-mean")
  )
  
  for (col in names(data)) {
    col_quantiles <- quantile(data[[col]], quantiles)
    col_stats <- c(min(data[[col]]), max(data[[col]]), col_quantiles, mean(data[[col]]), quantile(data[[col]], 0.995) - mean(data[[col]]))
    stats[[col]] <- col_stats
  }
  
  stats$Statystyki <- as.character(stats$Statystyki)
  stats$Statystyki[stats$Statystyki %in% quantiles] <- paste0("Q:", sprintf("%.1f%%", as.numeric(stats$Statystyki[stats$Statystyki %in% quantiles]) * 100))
  
  return(stats)
}

generate_pdf_report <- function(data_list, base_path) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M")
  save_path <- paste0(base_path, "_", timestamp, ".pdf")
  
  rmarkdown::render("templete.Rmd", output_file = save_path, params = list(data_list = data_list))
}

generate_report <- function(data_list, base_path = "RaportRyzykoPozaru") {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M")
  save_path <- paste0(base_path, "_", timestamp, ".docx")
  
  summary_data <- data.frame(
    Ubezpieczyciel = names(data_list),
    Brutto = numeric(length(data_list)),
    Brutto_Katastroficzne = numeric(length(data_list)),
    Netto = numeric(length(data_list)),
    Netto_Katastroficzne = numeric(length(data_list))
  )
  
  doc <- read_docx()
  
  liczba_ubezpieczycieli <- length(data_list)
  doc <- doc %>%
    body_add_par("RAPORT RYZYKO POŻARU", style = "heading 1") %>%
    body_add_par(Sys.Date(), style = "Normal") %>%
    body_add_par(paste0("Poniższy raport przedstawia analizę przeprowadzoną dla ", 
                        liczba_ubezpieczycieli, 
                        " ubezpieczycieli dla roku 2022. W rozdziale 2 zostały przedstawione wyniki SCR dla poszczególnych ubezpieczycieli. W kolejnych rozdziałach prezentowane są szczegółowe wyniki."), 
                 style = "Normal") %>%
    body_add_par("", style = "Normal")
  
  doc <- doc %>%
    body_add_par("Spis Treści", style = "heading 1") %>%
    body_add_toc(level = 2) %>%
    body_add_par("", style = "Normal")
  
  for (insurance_name in names(data_list)) {
    data <- data_list[[insurance_name]]
    stats <- calculate_statistics(data)
    
    cat("Processing:", insurance_name, "\n")
    print(stats)
    
    if ("0.995-mean" %in% stats$Statystyki) {
      brutto_value <- as.numeric(stats[stats$Statystyki == "0.995-mean", "Brutto"])
      brutto_kat_value <- as.numeric(stats[stats$Statystyki == "0.995-mean", "Brutto_Katastroficzny"])
      netto_value <- as.numeric(stats[stats$Statystyki == "0.995-mean", "Netto"])
      netto_kat_value <- as.numeric(stats[stats$Statystyki == "0.995-mean", "Netto_Katastroficzny"])
      
      summary_data[summary_data$Ubezpieczyciel == insurance_name, "Brutto"] <- brutto_value
      summary_data[summary_data$Ubezpieczyciel == insurance_name, "Brutto_Katastroficzne"] <- brutto_kat_value
      summary_data[summary_data$Ubezpieczyciel == insurance_name, "Netto"] <- netto_value
      summary_data[summary_data$Ubezpieczyciel == insurance_name, "Netto_Katastroficzne"] <- netto_kat_value
      
      cat("Assigned values for:", insurance_name, "\n")
      cat("Brutto:", brutto_value, "\n")
      cat("Brutto_Katastroficzne:", brutto_kat_value, "\n")
      cat("Netto:", netto_value, "\n")
      cat("Netto_Katastroficzne:", netto_kat_value, "\n")
    } else {
      cat("0.995-mean not found for:", insurance_name, "\n")
    }
  }
  
  doc <- doc %>%
    body_add_par("Podsumowanie", style = "heading 1") %>%
    body_add_table(value = summary_data, style = "table_template") %>%
    body_add_par("", style = "Normal")
  
  for (insurance_name in names(data_list)) {
    data <- data_list[[insurance_name]]
    stats <- calculate_statistics(data)
    
    doc <- doc %>%
      body_add_par(insurance_name, style = "heading 1") %>%
      body_add_par("Statystyki", style = "heading 2") %>%
      body_add_table(value = stats, style = "table_template") %>%
      body_add_par("Histogramy", style = "heading 2")
    
    for (col in names(data)) {
      hist_file <- tempfile(fileext = ".png")
      png(hist_file, width = 800, height = 600)
      hist(data[[col]], main = paste("Histogram", col), xlab = col, col = "blue", border = "black")
      dev.off()
      doc <- doc %>%
        body_add_par(col, style = "heading 3") %>%
        body_add_img(src = hist_file, width = 6, height = 4)
    }
    
    quantiles <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995)
    quantile_data <- data.frame(
      Quantile = rep(quantiles, times = ncol(data)),
      Value = unlist(lapply(data, quantile, quantiles)),
      Variable = rep(names(data), each = length(quantiles))
    )
    
    quantile_plot_file <- tempfile(fileext = ".png")
    png(quantile_plot_file, width = 800, height = 600)
    print(ggplot(quantile_data, aes(x = Quantile, y = Value, color = Variable)) +
            geom_line() +
            geom_point() +
            labs(title = "Kwantyle dla każdej kolumny", x = "Kwantyl", y = "Wartość szkody") +
            theme(legend.position = "left"))
    dev.off()
    
    doc <- doc %>%
      body_add_par("Kwantyle", style = "heading 2") %>%
      body_add_img(src = quantile_plot_file, width = 6, height = 4)
  }
  
  print(doc, target = save_path)
}

# Generowanie przykładowych danych
set.seed(123)
n <- 1000

data <- data.frame(
  Insurer = sample(0:5, n, replace = TRUE),
  Region = sample(0:15, n, replace = TRUE),
  Month = sample(0:11, n, replace = TRUE),
  SumValue = sample(1000:100000, n, replace = TRUE),
  IndexTable = 1:n,
  lat = runif(n, min = 49.0, max = 54.8),
  lon = runif(n, min = 14.1, max = 24.1)
)

generate_offset_coordinates <- function(lat, lon, distance) {
  bearing <- runif(1, 0, 360)
  new_coords <- destPoint(c(lon, lat), bearing, distance)
  return(new_coords)
}

set.seed(123)
sampled_buildings <- data %>% sample_n(100)

offset_data <- sampled_buildings %>%
  rowwise() %>%
  mutate(
    new_coords = list(generate_offset_coordinates(lat, lon, 200)),
    new_lat = new_coords[[2]],
    new_lon = new_coords[[1]],
    Region = Region,
    Month = Month
  ) %>%
  select(-new_coords) %>%
  ungroup()

new_buildings <- offset_data %>%
  mutate(
    Insurer = sample(0:5, n(), replace = TRUE),
    SumValue = sample(1000:100000, n(), replace = TRUE)
  ) %>%
  select(IndexTable, Insurer, Region, Month, SumValue, new_lat, new_lon)

colnames(new_buildings) <- c("IndexTable", "Insurer", "Region", "Month", "SumValue", "lat", "lon")

# UI
ui <- dashboardPage(
  dashboardHeader(title = "Analiza budynków według ubezpieczycieli"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Wizualizacja budynków", tabName = "viz_buildings", icon = icon("map")),
      menuItem("Symulacje", tabName = "simulations", icon = icon("chart-line"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "viz_buildings",
              fluidPage(
                fluidRow(
                  column(3,
                         selectInput("insurers", "Wybierz ubezpieczycieli do wyświetlenia:", 
                                     choices = unique(data$Insurer), multiple = TRUE),
                         selectInput("highlight", "Wybierz ubezpieczyciela do wyróżnienia:", 
                                     choices = unique(data$Insurer), selected = 0),
                         selectInput("regions", "Wybierz województwa do wyświetlenia:", 
                                     choices = unique(data$Region), multiple = TRUE),
                         selectInput("months", "Wybierz miesiące do wyświetlenia:", 
                                     choices = 0:11, multiple = TRUE),
                         numericInput("sumValueMin", "Minimalna suma ubezpieczenia:", value = NA, min = 0, step = 1000),
                         numericInput("sumValueMax", "Maksymalna suma ubezpieczenia:", value = NA, min = 0, step = 1000),
                         selectInput("mapTiles", "Wybierz typ mapy:", 
                                     choices = c("OpenStreetMap Mapnik" = "OpenStreetMap.Mapnik",
                                                 "Esri WorldStreetMap" = "Esri.WorldStreetMap",
                                                 "Esri WorldImagery" = "Esri.WorldImagery",
                                                 "OpenTopoMap" = "OpenTopoMap"))
                  ),
                  column(9,
                         box(width = 12, leafletOutput("map", height = "800px"))
                  )
                )
              )
      ),
      tabItem(tabName = "simulations",
              fluidPage(
                useShinyjs(),
                inlineCSS(list(
                  ".overlay" = "position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.5); z-index: 1000;",
                  ".spinner" = "position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 1001;"
                )),
                sidebarLayout(
                  sidebarPanel(
                    textInput("folder_path", "Ścieżka do folderu:", ""),
                    actionButton("load_data", "Pobierz dane"),
                    selectInput("insurance", "Wybierz ubezpieczyciela:", choices = NULL),
                    selectInput("column", "Wybierz kolumnę:", choices = NULL),
                    textInput("save_path", "Ścieżka do zapisu raportu:", ""),
                    selectInput("report_format", "Wybierz format raportu:", choices = c("Word", "PDF")),
                    actionButton("generate_report", "Generuj raport")
                  ),
                  mainPanel(
                    tabsetPanel(
                      id = "tabs",
                      tabPanel("Histogram",
                               h2("Histogram"),
                               withSpinner(plotOutput("histogramPlot"))
                      ),
                      tabPanel("Statystyki",
                               h2("Statystyki"),
                               withSpinner(DTOutput("statsTable"))
                      ),
                      tabPanel("Kwantyle",
                               h2("Kwantyle"),
                               withSpinner(plotOutput("quantilePlot"))
                      )
                    )
                  )
                ),
                div(id = "loadingOverlay", class = "overlay", style = "display: none;"),
                div(id = "loadingSpinner", class = "spinner", style = "display: none;", tags$img(src = "https://cdnjs.cloudflare.com/ajax/libs/timelinejs/2.36.0/css/loading.gif", height = "100"))
              )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  current_zoom <- reactiveVal(4)
  current_center <- reactiveVal(c(52, 19))
  
  observeEvent(input$mapTiles, {
    current_zoom(input$map_zoom)
    current_center(input$map_center)
  })
  
  output$map <- renderLeaflet({
    leaflet() %>%
      addProviderTiles(providers$OpenStreetMap.Mapnik) %>%
      setView(lng = 19, lat = 52, zoom = 4)
  })
  
  observeEvent(input$mapTiles, {
    proxy <- leafletProxy("map") %>%
      clearTiles() %>%
      addProviderTiles(switch(input$mapTiles,
                              "OpenStreetMap.Mapnik" = providers$OpenStreetMap.Mapnik,
                              "Esri.WorldStreetMap" = providers$Esri.WorldStreetMap,
                              "Esri.WorldImagery" = providers$Esri.WorldImagery,
                              "OpenTopoMap" = providers$OpenTopoMap)) %>%
      setView(lng = current_center()[2], lat = current_center()[1], zoom = current_zoom())
  })
  
  observe({
    leafletProxy("map") %>%
      clearMarkers() %>%
      clearShapes()
    
    if (is.null(input$insurers) || length(input$insurers) == 0) {
      filtered_data <- data
      filtered_new_buildings <- new_buildings
      filtered_sampled_buildings <- sampled_buildings
    } else {
      filtered_data <- data %>% filter(Insurer %in% input$insurers)
      filtered_new_buildings <- new_buildings %>% filter(Insurer %in% input$insurers)
      filtered_sampled_buildings <- sampled_buildings %>% filter(Insurer %in% input$insurers)
    }
    
    if (!is.null(input$regions) && length(input$regions) > 0) {
      filtered_data <- filtered_data %>% filter(Region %in% input$regions)
      filtered_new_buildings <- filtered_new_buildings %>% filter(Region %in% input$regions)
      filtered_sampled_buildings <- filtered_sampled_buildings %>% filter(Region %in% input$regions)
    }
    
    if (!is.null(input$months) && length(input$months) > 0) {
      filtered_data <- filtered_data %>% filter(Month %in% input$months)
      filtered_new_buildings <- filtered_new_buildings %>% filter(Month %in% input$months)
      filtered_sampled_buildings <- filtered_sampled_buildings %>% filter(Month %in% input$months)
    }
    
    if (!is.na(input$sumValueMin)) {
      filtered_data <- filtered_data %>% filter(SumValue >= input$sumValueMin)
      filtered_new_buildings <- filtered_new_buildings %>% filter(SumValue >= input$sumValueMin)
      filtered_sampled_buildings <- filtered_sampled_buildings %>% filter(SumValue >= input$sumValueMin)
    }
    if (!is.na(input$sumValueMax)) {
      filtered_data <- filtered_data %>% filter(SumValue <= input$sumValueMax)
      filtered_new_buildings <- filtered_new_buildings %>% filter(SumValue <= input$sumValueMax)
      filtered_sampled_buildings <- filtered_sampled_buildings %>% filter(SumValue <= input$sumValueMax)
    }
    
    leafletProxy("map") %>%
      addCircleMarkers(data = filtered_data,
                       ~lon, ~lat,
                       color = ~ifelse(Insurer == input$highlight, "red", "black"),
                       radius = ~log(SumValue) / 2,
                       popup = ~paste("Insurer:", Insurer, "<br>",
                                      "Region:", Region, "<br>",
                                      "Month:", Month + 1, "<br>",
                                      "SumValue:", SumValue, "<br>",
                                      "IndexTable:", IndexTable),
                       fillOpacity = 0.7,
                       group = "Original Buildings") %>%
      addCircleMarkers(data = filtered_new_buildings,
                       ~lon, ~lat,
                       color = ~ifelse(Insurer == input$highlight, "red", "blue"),
                       radius = 5,
                       popup = ~paste("Insurer:", Insurer, "<br>",
                                      "Region:", Region, "<br>",
                                      "Month:", Month + 1, "<br>",
                                      "SumValue:", SumValue, "<br>",
                                      "IndexTable:", IndexTable),
                       fillOpacity = 0.7,
                       group = "New Buildings") %>%
      addCircles(data = filtered_sampled_buildings,
                 ~lon, ~lat,
                 radius = 200,
                 color = "green",
                 fill = FALSE,
                 group = "Circles") %>%
      addLayersControl(
        overlayGroups = c("Original Buildings", "New Buildings", "Circles"),
        options = layersControlOptions(collapsed = FALSE)
      ) %>%
      addLegend(position = "bottomright",
                colors = c("black", "blue", "red", "green"),
                labels = c("Original Buildings", "New Buildings", "Highlighted Insurer", "Circles"),
                title = "Building Types") %>%
      addMiniMap(toggleDisplay = TRUE)
  })
  
  data_list <- reactiveVal(list())
  filtered_data <- reactiveVal(NULL)
  
  observeEvent(input$load_data, {
    shinyjs::show(id = "loadingOverlay")
    shinyjs::show(id = "loadingSpinner")
    
    folder_path <- input$folder_path
    if (dir.exists(folder_path)) {
      csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
      all_data <- lapply(csv_files, read.csv)
      names(all_data) <- gsub("\\.csv$", "", basename(csv_files))
      
      data_list(all_data)
      updateSelectInput(session, "insurance", choices = names(all_data))
      
      shinyjs::hide(id = "loadingOverlay")
      shinyjs::hide(id = "loadingSpinner")
    }
  })
  
  observeEvent(input$insurance, {
    req(input$insurance)
    filtered_data(data_list()[[input$insurance]])
    updateSelectInput(session, "column", choices = names(filtered_data()))
  })
  
  output$histogramPlot <- renderPlot({
    req(filtered_data())
    column <- input$column
    req(column)
    
    ggplot(filtered_data(), aes_string(x = column)) +
      geom_histogram(binwidth = 5000, fill = "blue", color = "black", alpha = 0.7) +
      labs(title = paste("Rozkład", column), x = column, y = "Częstotliwość")
  })
  
  output$statsTable <- renderDT({
    req(filtered_data())
    stats <- calculate_statistics(filtered_data())
    datatable(stats)
  })
  
  output$quantilePlot <- renderPlot({
    req(filtered_data())
    data <- filtered_data()
    quantiles <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995)
    
    quantile_data <- data.frame(
      Quantile = rep(quantiles, times = ncol(data)),
      Value = unlist(lapply(data, quantile, quantiles)),
      Variable = rep(names(data), each = length(quantiles))
    )
    
    ggplot(quantile_data, aes(x = Quantile, y = Value, color = Variable)) +
      geom_line() +
      geom_point() +
      labs(title = "Kwantyle dla każdej kolumny", x = "Kwantyl", y = "Wartość szkody") +
      theme(legend.position = "left")
  })
  
  observeEvent(input$generate_report, {
    req(data_list())
    req(input$save_path)
    
    report_format <- input$report_format
    
    if (report_format == "Word") {
      generate_report(data_list(), input$save_path)
    } else if (report_format == "PDF") {
      generate_pdf_report(data_list(), input$save_path)
    }
    
    showModal(modalDialog(
      title = "Raport wygenerowany",
      "Raport został pomyślnie zapisany w podanej lokalizacji.",
      easyClose = TRUE,
      footer = NULL
    ))
  })
}

shinyApp(ui = ui, server = server)
