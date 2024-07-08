import shinyswatch
import pandas as pd
import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui, run_app, reactive
from shiny.types import FileInfo
from shiny import experimental as x
import matplotlib.pyplot as plt
from shinywidgets import output_widget, render_widget
from shiny.types import ImgData
from metody_jednoroczne_copy import YearHorizont
# Dane
from metody_jednoroczne_copy import YearHorizont

yh = YearHorizont()
# Obliczenie ilorazów
def calculate_ratios(df):
    ratio_df = pd.DataFrame()
    for i in range(1, df.shape[1] - 1):
        ratio_col = df.iloc[:, i + 1] / df.iloc[:, i]
        ratio_df[f'Ratio_{i + 1}/{i}'] = ratio_col
    return ratio_df

# Funkcja do tworzenia DataFrame wypełnionego 1 i NaN
def create_binary_df(ratio_df):
    binary_df = ratio_df.applymap(lambda x: 1 if pd.notna(x) else np.nan)
    return binary_df

# JavaScript code to handle cell edits and clicks
js_code_p = """
$(document).on('click', '#ratios-table-1 td', function() {
    var row = $(this).closest('tr').index();
    var col = $(this).index();
    if ($(this).hasClass('highlighted')) {
        $(this).removeClass('highlighted');
        Shiny.setInputValue('clicked_cell_ratios_table_1', {row: row, col: col - 1, highlighted: false});
    } else {
        $(this).addClass('highlighted');
        Shiny.setInputValue('clicked_cell_ratios_table_1', {row: row, col: col - 1, highlighted: true});
    }
});
"""

js_code_i = """
$(document).on('click', '#ratios-table-2 td', function() {
    var row = $(this).closest('tr').index();
    var col = $(this).index();
    if ($(this).hasClass('highlighted')) {
        $(this).removeClass('highlighted');
        Shiny.setInputValue('clicked_cell_ratios_table_2', {row: row, col: col - 1, highlighted: false});
    } else {
        $(this).addClass('highlighted');
        Shiny.setInputValue('clicked_cell_ratios_table_2', {row: row, col: col - 1, highlighted: true});
    }
});
"""

# CSS for highlighted cells
css_code = """
.highlighted {
    background-color: yellow !important;
}
"""

# Definiowanie interfejsu użytkownika
app_ui = ui.page_fluid(
    ui.navset_tab(

ui.nav("Wczytaj dane",
       ui.input_file("file", "Wybierz Excel z danymi", accept=[".xlsx"]),
       ui.input_select("sheet_name", "Wybierz linię biznesową", {i: str(i) for i in range(1, 25)}, selected=1),
       ui.input_numeric("start_row", "Podaj wiersz od którego wczytać dane", value=5, min=1),
       ui.input_numeric("num_rows", "Podaj ilośc wierszy", value=34, min=1),
       ui.input_text("usecols", "Podaj kolumny, z których wczytać dane", value="A:AI"),
       ui.input_action_button("load_data", "Wczytaj")

       ),

        ui.nav("Paid Claims",
ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.input_selectize("linie_biznesowe_CL_Paid", "Wybierz linię biznesową", choices=['-'], multiple=False),
                    ui.input_numeric("ilosc_okresow", "Ilość okresów", value=0),
                    x.ui.accordion(
                        x.ui.accordion_panel(
                            "Dopasowanie CL",
                            ui.input_numeric("x", "Maksymalna wartośc CL", value=3),
                            ui.input_numeric("Poz_CL", "Pozostawione CL", value=0),
                            ui.input_numeric("Max_CL", "Maksymalny CL", value=10),
                            ui.input_numeric("Min_CL", "Minimlany CL", value=1),
                            ui.input_selectize('chose_CL', 'Wybierz CL do dopasowania krzywej',
                                               [int(x) for x in range(1, 20)], selected=[1, 2], multiple=True),
                            ui.input_action_button("accept_CL", "Dopasuj krzywą", class_="btn-success"),
                        ),
                        x.ui.accordion_panel(
                            "Dopasowanie wariancji CL",
                            ui.input_numeric("loss_max_var", "Maksymalna wartośc wariancji", value=100000),
                            ui.input_numeric("Poz_CL_var", "Pozostawione wariancji", value=2),
                            ui.input_numeric("Max_var", "Maksymalna wariancja", value=1000000),
                            ui.input_numeric("Min_var", "Minimlana wariancja", value=0),
                            ui.input_selectize('chose_var', 'Wybierz wariancje do dopasowania krzywej',
                                               [int(x) for x in range(1, 20)], selected=[1, 2], multiple=True),
                            ui.input_action_button("accept_CL_var", "Dopasuj krzywą", class_="btn-success"),
                        ),
                        id='id_panel', open=False, multiple=False
                    ),
                    width=2,
                ),
    ui.panel_main(
            ui.navset_tab(
                ui.nav_panel("Trójkąt",
                    ui.input_slider("height_p", "Height of the triangle", min=1, max=20, value=5),
                    ui.output_table("triangle_table_p")
                ),
                ui.nav_panel("Ilorazy",
                    ui.output_ui("ratios_table_ui_p")
                ),
                ui.nav_panel("Binary Ilorazy",
                    ui.output_ui("binary_ratios_table_ui_p")
                ),
                ui.nav(
                    "Skumulowane CL",
                    x.ui.page_fillable(
                        x.ui.layout_column_wrap(1, x.ui.card(ui.output_data_frame("macierz_wspol_CL_interaktywna")),
                                                height=180)
                    ),
                    x.ui.page_fillable(
                        x.ui.layout_column_wrap(1,
                                                x.ui.card(ui.output_data_frame("wspol_z_krzywej_CL_paid_interaktywna")),
                                                height=180)
                    ),
                    x.ui.layout_column_wrap(
                        1,
                        x.ui.card(ui.output_plot("plot_wspolczynniki_dopasowane_interaktywny")),
                        height=400
                    ),
                ),
                ui.nav(
                    "Wizualizacja i wyniki",
                    x.ui.page_fillable(
                        x.ui.layout_column_wrap(1, x.ui.card(ui.output_data_frame("Ult_BE_data_interaktywne")),
                                                height=400)
                    ),
                ),
            ),
            ui.tags.style(css_code),
            ui.tags.script(js_code_p)
        ))
               ),
        ui.nav("Incurred Claims"
               ,
               ui.layout_sidebar(
                   ui.panel_sidebar(
                       ui.input_selectize("linie_biznesowe_CL__incurred", "Wybierz linię biznesową", choices=['-'],
                                          multiple=False),
                       ui.input_numeric("ilosc_okresow_incurred", "Ilość okresów", value=0),
                       x.ui.accordion(
                           x.ui.accordion_panel(
                               "Dopasowanie CL",
                               ui.input_numeric("x_incurred", "Maksymalna wartośc CL", value=3),
                               ui.input_numeric("Poz_CL_incurred", "Pozostawione CL", value=2),
                               ui.input_numeric("Max_CL_incurred", "Maksymalny CL", value=10),
                               ui.input_numeric("Min_CL_incurred", "Minimlany CL", value=1),
                               ui.input_selectize('chose_CL_incurred', 'Wybierz CL do dopasowania krzywej',
                                                  [int(x) for x in range(1, 20)], selected=[1, 2], multiple=True),
                               ui.input_action_button("accept_CL_incurred", "Dopasuj krzywą", class_="btn-success"),
                           ),
                           x.ui.accordion_panel(
                               "Dopasowanie wariancji CL",
                               ui.input_numeric("loss_max_var_incurred", "Maksymalna wartośc wariancji", value=100000),
                               ui.input_numeric("Poz_CL_var_incurred", "Pozostawione wariancji", value=0),
                               ui.input_numeric("Max_var_incurred", "Maksymalna wariancja", value=1000000),
                               ui.input_numeric("Min_var_incurred", "Minimlana wariancja", value=0),
                               ui.input_selectize('chose_var_incurred', 'Wybierz wariancje do dopasowania krzywej',
                                                  [int(x) for x in range(1, 20)], selected=[1, 2], multiple=True),
                               ui.input_action_button("accept_CL_var_incurred", "Dopasuj krzywą", class_="btn-success"),
                           ),
                           id='id_panel_incurred', open=False, multiple=False
                       ),
                       width=2,
                   ),
                   ui.panel_main(
                       ui.navset_tab(
                           ui.nav_panel("Trójkąt",
                                        ui.input_slider("height_p", "Height of the triangle", min=1, max=20, value=5),
                                        ui.output_text("triangle_i"),
                                        ui.output_table("triangle_table_i")
                                        ),
                           ui.nav_panel("Ilorazy",
                                        ui.output_ui("ratios_table_ui_i")
                                        ),
                           ui.nav_panel("Binary Ilorazy",
                                        ui.output_ui("binary_ratios_table_ui_i")
                                        ),
                           ui.nav(
                               "Skumulowane CL",
                               x.ui.page_fillable(
                                   x.ui.layout_column_wrap(1, x.ui.card(
                                       ui.output_data_frame("macierz_wspol_CL_interaktywna_incurred")),
                                                           height=180)
                               ),
                               x.ui.page_fillable(
                                   x.ui.layout_column_wrap(1, x.ui.card(
                                       ui.output_data_frame("wspol_z_krzywej_CL_interaktywna_incurred")),
                                                           height=180)
                               ),
                               x.ui.layout_column_wrap(
                                   1,
                                   x.ui.card(ui.output_plot("plot_wspolczynniki_dopasowane_interaktywny_incurred")),
                                   height=400
                               ),
                           ),
                           ui.nav(
                               "Wizualizacja i wyniki",
                               x.ui.page_fillable(
                                   x.ui.layout_column_wrap(1,
                                                           x.ui.card(ui.output_data_frame("Ult_BE_data_interaktywne_incurred")),
                                                           height=400)
                               ),
                           ),
                       ),
                       ui.tags.style(css_code),
                       ui.tags.script(js_code_i)
                   ))
               ),
        )
    )


# Definiowanie funkcji serwera
def server(input: Inputs, output: Outputs, session: Session):
    df_paid = reactive.Value(pd.DataFrame())
    df_incurred = reactive.Value(pd.DataFrame())
    @reactive.Effect
    #@reactive.event(input.load_data)
    def load_data():
        file = input.file()
        sheet_name_paid = f"DFM paid ({input.sheet_name()})"
        sheet_name_incurred = f"DFM inccured ({input.sheet_name()})"
        start_row = input.start_row() - 1  # Korekta dla indeksowania 0
        num_rows = input.num_rows()
        usecols = input.usecols()
        print(sheet_name_paid)
        print(sheet_name_incurred)
        if not file:
            df_paid.set(pd.DataFrame())
            df_incurred.set(pd.DataFrame())
        try:
            df_paid.set(
                pd.read_excel(file[0]['datapath'], sheet_name=sheet_name_paid, usecols=usecols, skiprows=start_row,
                              nrows=num_rows))
            df_incurred.set(
                pd.read_excel(file[0]['datapath'], sheet_name=sheet_name_incurred, usecols=usecols, skiprows=start_row,
                              nrows=num_rows))
        except Exception as e:
            print(f"Błąd wczytywania danych: {e}")
            df_paid.set(pd.DataFrame())
            df_incurred.set(pd.DataFrame())

    @reactive.Calc
    def triangle_paid():
        df = df_paid.get()
        # Dodaj tu logikę przetwarzania dla triangle_paid
        return df  # Zwróć przetworzone dane

    @reactive.Calc
    def triangle_incurred():
        df = df_incurred.get()
        # Dodaj tu logikę przetwarzania dla triangle_incurred
        return df  # Zwróć przetworzone dane
####################################################
    # Zakładka P
    clicked_cells_p = reactive.Value([])
    update_trigger_p = reactive.Value(0)

    ratio_df_p = calculate_ratios(triangle_paid())
    binary_df_p = create_binary_df(ratio_df_p)

    @output
    @render.table
    def triangle_table_p():
        return triangle_paid()

    @output
    @render.ui
    def ratios_table_ui_p():
        return ui.HTML(ratio_df_p.to_html(classes='table table-striped table-hover', table_id="ratios-table-1"))

    @output
    @render.ui
    def binary_ratios_table_ui_p():
        update_trigger_p.get()
        df = binary_df_p.copy()
        return ui.HTML(df.to_html(classes='table table-striped table-hover', table_id="binary-ratios-table-1", na_rep='NaN', float_format='{:.0f}'.format))

    @reactive.Effect
    @reactive.event(input.clicked_cell_ratios_table_1)
    def update_clicked_cell_p():
        cell = input.clicked_cell_ratios_table_1()
        print(f"Cell clicked in ratios table for P: {cell}")  # Debug print
        if cell:
            row, col, highlighted = cell['row'], cell['col'], cell['highlighted']
            if highlighted:
                binary_df_p.iat[row, col] = 0
            else:
                binary_df_p.iat[row, col] = 1
            update_trigger_p.set(update_trigger_p.get() + 1)

    # Zakładka I
    clicked_cells_i = reactive.Value([])
    update_trigger_i = reactive.Value(0)

    ratio_df_i = calculate_ratios(triangle_incurred())
    binary_df_i = create_binary_df(ratio_df_i)

    @output
    @render.text
    def triangle_i():
        height = input.height_i()
        return '\n'.join(' ' * (height - i - 1) + '*' * (2 * i + 1) for i in range(height))

    @output
    @render.table
    def triangle_table_i():
        return triangle_incurred()

    @output
    @render.ui
    def ratios_table_ui_i():
        return ui.HTML(ratio_df_i.to_html(classes='table table-striped table-hover', table_id="ratios-table-2"))

    @output
    @render.ui
    def binary_ratios_table_ui_i():
        update_trigger_i.get()
        df = binary_df_i.copy()
        return ui.HTML(df.to_html(classes='table table-striped table-hover', table_id="binary-ratios-table-2", na_rep='NaN', float_format='{:.0f}'.format))

    @reactive.Effect
    @reactive.event(input.clicked_cell_ratios_table_2)
    def update_clicked_cell_i():
        cell = input.clicked_cell_ratios_table_2()
        if cell:
            row, col, highlighted = cell['row'], cell['col'], cell['highlighted']
            if highlighted:
                binary_df_i.iat[row, col] = 0
            else:
                binary_df_i.iat[row, col] = 1
            update_trigger_i.set(update_trigger_i.get() + 1)

    ### Zakładki ze współczynnikami
    ###########################################################################################
    #Paid
    @reactive.event(input.clicked_cell_ratios_table_1)
    def wspolczynniki_multiplikatywna_interaktywna():
        triagnle = triangle_paid().iloc[:, 1:]
        binary_df_pd = binary_df_p.copy()
        binary_df_deterministic = yh.create_binary_df(triagnle)
        ind_all, m_i, m_first = yh.index_all(triagnle)
        macierz_wsp_l = yh.l_i_j(triagnle, ind_all)
        print("Dev_j_deterministic")
        Dev_j_deterministic = yh.Dev(triagnle, binary_df_deterministic, macierz_wsp_l, ind_all)
        print("Dev_j")
        Dev_j = yh.Dev(triagnle, binary_df_pd, macierz_wsp_l, ind_all)
        sigma_j = yh.sigma(triagnle, binary_df_pd, macierz_wsp_l, Dev_j, ind_all)
        sd_j = yh.wspolczynnik_sd(triagnle, binary_df_pd, sigma_j, ind_all)
        I_dataframe = pd.DataFrame(0, index=['CL_base', 'CL', 'sigma', 'sd'],
                                   columns=[str(x) for x in range(1, len(Dev_j) + 2)])
        I_dataframe.iloc[0, :] = ["CL_base"] + Dev_j_deterministic
        I_dataframe.iloc[1, :] = ["CL"] + Dev_j
        I_dataframe.iloc[2, :] = ["sigma"] + sigma_j
        I_dataframe.iloc[3, :] = ["sd"] + sd_j
        return I_dataframe

    @output
    @render.data_frame
    def macierz_wspol_CL_interaktywna():
        df_out_mult = wspolczynniki_multiplikatywna_interaktywna()
        return render.DataGrid(
            df_out_mult,
            width="100%",
            height="150%",
        )

    @reactive.Calc
    @reactive.event(input.accept_CL, ignore_none=False)
    def dopasowanie_krzywej_factor_interaktywne():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[1, 1:]
        sd_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[3, 1:]
        ilosc_dop_wsp_CL = [int(x) for x in input.chose_CL()]
        vector_value, x_k_ind = yh.check_value(Dev_pd, ilosc_dop_wsp_CL,
                                               input.Min_CL(), input.Max_CL())
        sd_chose = [sd_pd[x] for x in x_k_ind]
        n_CL = len(Dev_pd) - (len(ilosc_dop_wsp_CL) - len(x_k_ind))
        a, b = yh.fit_curve(vector_value, sd_chose, x_k_ind, 'factor_CL', n_CL)
        dev_pozostawione = [Dev_pd[x] for x in range(0, (input.Poz_CL()))]
        print(dev_pozostawione)
        print(len(Dev_pd.tolist()))
        vec_output = ['CL'] + dev_pozostawione + yh.wspolczynnik_reg(a, b, input.Poz_CL() + 1,
                                                                     len(Dev_pd.tolist()) + input.ilosc_okresow(),
                                                                     'factor_CL')
        return (vec_output)

    @reactive.Calc
    @reactive.event(input.accept_CL_var or input.accept_CL, ignore_none=False)
    def dopasowanie_krzywej_variance_interaktywne():
        sigma_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[2, 1:]
        sd_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[3, 1:]
        ilosc_dop_wsp_var = [int(x) for x in input.chose_var()]
        sigma_pd_choose = [sigma_pd[x] for x in ilosc_dop_wsp_var]
        vector_value, x_k_ind = yh.check_value(sigma_pd_choose, ilosc_dop_wsp_var,
                                               input.Min_var(), input.Max_var())
        n_CL_var = len(sigma_pd) - (len(ilosc_dop_wsp_var) - len(x_k_ind))
        sd_chose = [sd_pd[x] for x in x_k_ind]
        a, b = yh.fit_curve(vector_value, sd_chose, x_k_ind, 'variance_CL', n_CL_var)
        sigma_pozostawione = [sigma_pd[x] for x in range(0, (input.Poz_CL_var()))]
        vec_output = ['sigma'] + sigma_pozostawione + yh.wspolczynnik_reg(a, b, input.Poz_CL_var() + 1,
                                                                          len(sigma_pd.tolist()) + input.ilosc_okresow(),
                                                                          'variance_CL')
        return (vec_output)

    @output
    @render.data_frame
    def wspol_z_krzywej_CL_paid_interaktywna():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[0,:]
        size_col = len(Dev_pd)+input.ilosc_okresow()
        II_dataframe = pd.DataFrame(0, index=[0, 1, 2], columns=[str(x) for x in range(1, size_col + 1)])
        print(II_dataframe.to_string())
        if len(dopasowanie_krzywej_factor_interaktywne()) != size_col and len(dopasowanie_krzywej_variance_interaktywne()) == size_col:
            II_dataframe.iloc[1, :] = dopasowanie_krzywej_variance_interaktywne()
        elif len(dopasowanie_krzywej_variance_interaktywne()) != size_col and len(dopasowanie_krzywej_factor_interaktywne()) == (size_col):
            II_dataframe.iloc[0, :] = dopasowanie_krzywej_factor_interaktywne()
        elif len(dopasowanie_krzywej_variance_interaktywne()) == size_col and len(dopasowanie_krzywej_factor_interaktywne()) == size_col:
            II_dataframe.iloc[0, :] = dopasowanie_krzywej_factor_interaktywne()
            II_dataframe.iloc[1, :] = dopasowanie_krzywej_variance_interaktywne()
        return render.DataGrid(
            II_dataframe,
            width="100%",
            height="100%",
            row_selection_mode='single'
        )

    @output
    @render.plot()
    def plot_wspolczynniki_dopasowane_interaktywny():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[1, 1:]
        sigma_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[2, 1:]
        fig = plt.figure()
        if input.id_panel() is None:
            plt.xticks(np.arange(1, len(sigma_pd.tolist()) + input.ilosc_okresow()))
            fig.autofmt_xdate()
        elif input.id_panel()[0] == 'Dopasowanie wariancji CL':
            var_fit = dopasowanie_krzywej_variance_interaktywne()
            plt.plot(np.arange(1, len(sigma_pd.tolist()) + 1), sigma_pd.to_list(), 'b', label='Sigma CL')
            plt.plot(np.arange(input.Poz_CL_var(), len(var_fit)), var_fit[input.Poz_CL_var():], 'r',
                     label='Dopasowana Sigma CL')
            plt.xticks(np.arange(1, len(sigma_pd.tolist()) + 1 + input.ilosc_okresow()))
            fig.autofmt_xdate()
            fig.legend()
        elif input.id_panel()[0] == 'Dopasowanie CL':
            CL_fit = dopasowanie_krzywej_factor_interaktywne()
            plt.plot(np.arange(1, len(Dev_pd.tolist()) + 1), Dev_pd.to_list(), 'b', label='CL')
            plt.plot(np.arange(input.Poz_CL(), len(CL_fit)), CL_fit[input.Poz_CL():], 'r', label='Dopasowane CL')
            plt.xticks(np.arange(1, len(Dev_pd.tolist()) + 1 + input.ilosc_okresow()))
            fig.autofmt_xdate()
            fig.legend()
        return fig

    @reactive.Calc
    def calc_chainladder_interaktywne():
        triangle = triangle_paid().iloc[:, 1:]
        CL_fit = dopasowanie_krzywej_factor_interaktywne()
        Dev_j_base = wspolczynniki_multiplikatywna_interaktywna().iloc[0, 1:].to_list()
        Dev_j_z_wagami = wspolczynniki_multiplikatywna_interaktywna().iloc[1, 1:].to_list()
        data_output = pd.DataFrame(0, index=np.arange(0, triangle.shape[0] + 1),
                                   columns=['Rok/Suma', 'Ult_base', 'IBNR_base', 'Ult z wagami', 'IBNR z wagami',
                                            'Ult z krzywą', 'IBNR z krzywą'])
        data_output.iloc[:, 0] = np.arange(0, triangle.shape[0] + 1)
        k = 1
        for wspolczynniki in [Dev_j_base, Dev_j_z_wagami, CL_fit[1:]]:
            proj_triangle = yh.triangle_forward(triangle, wspolczynniki, 0)
            diag = yh.reverse_list(yh.trian_diag(triangle))
            Ultimate_Param_ReservingRisk = proj_triangle.iloc[:, int(proj_triangle.columns[-1]) - 1].to_list()
            data_output.iloc[:, k] = Ultimate_Param_ReservingRisk + [np.sum(Ultimate_Param_ReservingRisk)]
            k = k + 1
            BE_Param_ReservingRisk = [x - y for x, y in zip(Ultimate_Param_ReservingRisk, diag)]
            data_output.iloc[:, k] = BE_Param_ReservingRisk + [np.sum(BE_Param_ReservingRisk)]
            k = k + 1
        return (data_output)

    @output
    @render.data_frame
    def Ult_BE_data_interaktywne():
        df = calc_chainladder_interaktywne()
        return render.DataGrid(
            df,
            width="100%",
            height="150%",
        )
#########################################################################################
#Incurred
    @reactive.event(input.clicked_cell_ratios_table_2)
    def wspolczynniki_multiplikatywna_interaktywna_incurred():
        triagnle = triangle_incurred().iloc[:, 1:]
        binary_df_pd = binary_df_i.copy()
        binary_df_deterministic = yh.create_binary_df(triagnle)
        ind_all, m_i, m_first = yh.index_all(triagnle)
        macierz_wsp_l = yh.l_i_j(triagnle, ind_all)
        Dev_j_deterministic = yh.Dev(triagnle, binary_df_deterministic, macierz_wsp_l, ind_all)
        Dev_j = yh.Dev(triagnle, binary_df_pd, macierz_wsp_l, ind_all)
        sigma_j = yh.sigma(triagnle, binary_df_pd, macierz_wsp_l, Dev_j, ind_all)
        sd_j = yh.wspolczynnik_sd(triagnle, binary_df_pd, sigma_j, ind_all)
        I_dataframe = pd.DataFrame(0, index=['CL_base', 'CL', 'sigma', 'sd'],
                                   columns=[str(x) for x in range(1, len(Dev_j) + 2)])
        I_dataframe.iloc[0, :] = ["CL_base"] + Dev_j_deterministic
        I_dataframe.iloc[1, :] = ["CL"] + Dev_j
        I_dataframe.iloc[2, :] = ["sigma"] + sigma_j
        I_dataframe.iloc[3, :] = ["sd"] + sd_j
        return I_dataframe

    @output
    @render.data_frame
    def macierz_wspol_CL_interaktywna_incurred():
        df_out_mult = wspolczynniki_multiplikatywna_interaktywna_incurred()
        return render.DataGrid(
            df_out_mult,
            width="100%",
            height="150%",
        )
##
    @reactive.Calc
    @reactive.event(input.accept_CL_incurred, ignore_none=False)
    def dopasowanie_krzywej_factor_interaktywne_incurred():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[1, 1:]
        sd_pd = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[3, 1:]
        ilosc_dop_wsp_CL = [int(x) for x in input.chose_CL_incurred()]
        vector_value, x_k_ind = yh.check_value(Dev_pd, ilosc_dop_wsp_CL,
                                               input.Min_CL_incurred(), input.Max_CL_incurred())
        sd_chose = [sd_pd[x] for x in x_k_ind]
        n_CL = len(Dev_pd) - (len(ilosc_dop_wsp_CL) - len(x_k_ind))
        a, b = yh.fit_curve(vector_value, sd_chose, x_k_ind, 'factor_CL', n_CL)
        dev_pozostawione = [Dev_pd[x] for x in range(0, (input.Poz_CL_incurred()))]
        vec_output = ['CL'] + dev_pozostawione + yh.wspolczynnik_reg(a, b, input.Poz_CL_incurred() + 1,
                                                                     len(Dev_pd.tolist()) + input.ilosc_okresow_incurred(),
                                                                     'factor_CL')
        return (vec_output)

    @reactive.Calc
    @reactive.event(input.accept_CL_var_incurred, ignore_none=False)
    def dopasowanie_krzywej_variance_interaktywne_incurred():
        sigma_pd = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[2, 1:]
        sd_pd = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[3, 1:]
        ilosc_dop_wsp_var = [int(x) for x in input.chose_var_incurred()]
        sigma_pd_choose = [sigma_pd[x] for x in ilosc_dop_wsp_var]
        vector_value, x_k_ind = yh.check_value(sigma_pd_choose, ilosc_dop_wsp_var,
                                               input.Min_var_incurred(), input.Max_var_incurred())
        n_CL_var = len(sigma_pd) - (len(ilosc_dop_wsp_var) - len(x_k_ind))
        sd_chose = [sd_pd[x] for x in x_k_ind]
        a, b = yh.fit_curve(vector_value, sd_chose, x_k_ind, 'variance_CL', n_CL_var)
        sigma_pozostawione = [sigma_pd[x] for x in range(0, (input.Poz_CL_var()))]
        vec_output = ['sigma'] + sigma_pozostawione + yh.wspolczynnik_reg(a, b, input.Poz_CL_var_incurred() + 1,
                                                                          len(sigma_pd.tolist()) + input.ilosc_okresow_incurred(),
                                                                          'variance_CL')
        return (vec_output)

    @output
    @render.data_frame
    def wspol_z_krzywej_CL_interaktywna_incurred():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[0, :]
        size_col = len(Dev_pd) + input.ilosc_okresow_incurred()
        II_dataframe = pd.DataFrame(0, index=[0, 1, 2], columns=[str(x) for x in range(1, size_col + 1)])
        if len(dopasowanie_krzywej_factor_interaktywne_incurred()) != size_col and len(dopasowanie_krzywej_variance_interaktywne_incurred()) == size_col:
            II_dataframe.iloc[1, :] = dopasowanie_krzywej_variance_interaktywne_incurred()
        elif len(dopasowanie_krzywej_variance_interaktywne_incurred()) != size_col and len(dopasowanie_krzywej_factor_interaktywne_incurred()) == (size_col):
            II_dataframe.iloc[0, :] = dopasowanie_krzywej_factor_interaktywne_incurred()
        elif len(dopasowanie_krzywej_variance_interaktywne_incurred()) == size_col and len(dopasowanie_krzywej_factor_interaktywne_incurred()) == size_col:
            II_dataframe.iloc[0, :] = dopasowanie_krzywej_factor_interaktywne_incurred()
            II_dataframe.iloc[1, :] = dopasowanie_krzywej_variance_interaktywne_incurred()
        return render.DataGrid(
            II_dataframe,
            width="100%",
            height="100%",
            row_selection_mode='single'
        )

    @output
    @render.plot()
    def plot_wspolczynniki_dopasowane_interaktywny_incurred():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[1, 1:]
        sigma_pd = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[2, 1:]
        fig = plt.figure()
        if input.id_panel_incurred() is None:
            plt.xticks(np.arange(1, len(sigma_pd.tolist()) + input.ilosc_okresow_incurred()))
            fig.autofmt_xdate()
        elif input.id_panel_incurred()[0] == 'Dopasowanie wariancji CL':
            var_fit = dopasowanie_krzywej_variance_interaktywne_incurred()
            plt.plot(np.arange(1, len(sigma_pd.tolist()) + 1), sigma_pd.to_list(), 'b', label='Sigma CL')
            plt.plot(np.arange(input.Poz_CL_var_incurred(), len(var_fit)), var_fit[input.Poz_CL_var_incurred():], 'r', label='Dopasowana Sigma CL')
            plt.xticks(np.arange(1, len(sigma_pd.tolist()) + 1 + input.ilosc_okresow_incurred()))
            fig.autofmt_xdate()
            fig.legend()
        elif input.id_panel_incurred()[0] == 'Dopasowanie CL':
            CL_fit = dopasowanie_krzywej_factor_interaktywne_incurred()
            plt.plot(np.arange(1, len(Dev_pd.tolist()) + 1), Dev_pd.to_list(), 'b', label='CL')
            plt.plot(np.arange(input.Poz_CL_incurred(), len(CL_fit)), CL_fit[input.Poz_CL_incurred():], 'r', label='Dopasowane CL')
            plt.xticks(np.arange(1, len(Dev_pd.tolist()) + 1 + input.ilosc_okresow_incurred()))
            fig.autofmt_xdate()
            fig.legend()
        return fig

    @reactive.Calc
    def calc_chainladder_interaktywne_incurred():
        triangle = triangle_incurred().iloc[:, 1:]
        CL_fit = dopasowanie_krzywej_factor_interaktywne_incurred()
        Dev_j_base = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[0, 1:].to_list()
        Dev_j_z_wagami = wspolczynniki_multiplikatywna_interaktywna_incurred().iloc[1, 1:].to_list()
        data_output = pd.DataFrame(0, index=np.arange(0, triangle.shape[0] + 1),
                                   columns=['Rok/Suma', 'Ult_base', 'IBNR_base', 'Ult z wagami', 'IBNR z wagami',
                                            'Ult z krzywą', 'IBNR z krzywą'])
        data_output.iloc[:, 0] = np.arange(0, triangle.shape[0] + 1)
        k = 1
        for wspolczynniki in [Dev_j_base, Dev_j_z_wagami, CL_fit[1:]]:
            proj_triangle = yh.triangle_forward(triangle, wspolczynniki, 0)
            diag = yh.reverse_list(yh.trian_diag(triangle))
            Ultimate_Param_ReservingRisk = proj_triangle.iloc[:, int(proj_triangle.columns[-1]) - 1].to_list()
            data_output.iloc[:, k] = Ultimate_Param_ReservingRisk + [np.sum(Ultimate_Param_ReservingRisk)]
            k = k + 1
            BE_Param_ReservingRisk = [x - y for x, y in zip(Ultimate_Param_ReservingRisk, diag)]
            data_output.iloc[:, k] = BE_Param_ReservingRisk + [np.sum(BE_Param_ReservingRisk)]
            k = k + 1
        return (data_output)

    @output
    @render.data_frame
    def Ult_BE_data_interaktywne_incurred():
        df = calc_chainladder_interaktywne_incurred()
        return render.DataGrid(
            df,
            width="100%",
            height="150%",
        )



# Tworzenie aplikacji
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()