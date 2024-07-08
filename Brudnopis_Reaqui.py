import shinyswatch
import pandas as pd
import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui,run_app,reactive
from shiny.types import FileInfo
from shiny import experimental as x
import matplotlib.pyplot as plt
from shinywidgets import output_widget, render_widget
from shiny.types import ImgData


# Obliczenie ilorazów



# JavaScript code to handle cell edits and clicks
js_code = """
$(document).on('click', 'td', function() {
    var row = $(this).closest('tr').index();
    var col = $(this).index();
    if ($(this).hasClass('highlighted')) {
        $(this).removeClass('highlighted');
        Shiny.setInputValue('clicked_cell', {row: row, col: col - 1, highlighted: false});
    } else {
        $(this).addClass('highlighted');
        Shiny.setInputValue('clicked_cell', {row: row, col: col - 1, highlighted: true});
    }
});
"""

# CSS for highlighted cells
css_code = """
.highlighted {
    background-color: gray !important;
}
"""

js_code_incurred = """
$(document).on('click', 'td', function() {
    var row = $(this).closest('tr').index();
    var col = $(this).index();
    if ($(this).hasClass('highlighted')) {
        $(this).removeClass('highlighted');
        Shiny.setInputValue('clicked_cell', {row: row, col: col - 1, highlighted: false});
    } else {
        $(this).addClass('highlighted');
        Shiny.setInputValue('clicked_cell', {row: row, col: col - 1, highlighted: true});
    }
});
"""

# CSS for highlighted cells
css_code_incurred = """
.highlighted {
    background-color: gray !important;
}
"""

from metody_jednoroczne_copy import YearHorizont
yh = YearHorizont()
app_ui = ui.page_fluid(ui.page_navbar(
    shinyswatch.theme.superhero(),
    ui.nav("Igloo", ui.output_image("image")),
    ui.nav("Wprowadź dane",
        ui.row(
            ui.column(
                4,
                "Dane do rezerw",
                ui.input_file("file1", "Wprowadź listę trójkątów w Excel", accept=[".xlsx"], multiple=False),
                ui.input_file("wagi_input", "Wprowadź wagi dla CL", accept=[".xlsx"], multiple=False),
                ui.input_file("wagi_input_LR", "Wprowadź wagi dla LR", accept=[".xlsx"], multiple=False),
                ui.input_file("ekspozycja_input", "Wprowadź ekspozycję w Excel", accept=[".xlsx"], multiple=False),
                ui.input_file("inflacja_input", "Wprowadź inflację w Excel", accept=[".xlsx"], multiple=False),
                ui.input_switch("potwierdz_inflacje", "Uwzględnij inflację"),
                ui.input_action_button("go", "Wykonaj obliczenia", class_="btn-success"),
                x.ui.card(ui.output_text_verbatim("wykonaj_funkcje"))
            ),
            ui.column(
                4,
                "Wagi P/I",
                ui.input_file("wagi_p_i_input", "Wprowadź wagi dla P/I", accept=[".xlsx"], multiple=False),
            ),
        ),

           ),
    ui.nav("Paid Claims",ui.layout_sidebar(ui.panel_sidebar(ui.input_selectize("linie_biznesowe_CL_Paid", "Wybierz linię biznesową", choices=['-'], multiple=False),
                                                             ui.input_numeric("ilosc_okresow", "Ilość okresów",
                                                                              value=0),
                                                             x.ui.accordion(x.ui.accordion_panel(
                                              "Dopasowanie CL",
                                              ui.input_numeric("x", "Maksymalna wartośc CL", value=3),
                                              ui.input_numeric("Poz_CL", "Pozostawione CL", value=2),
                                              ui.input_numeric("Max_CL", "Maksymalny CL", value=10),
                                              ui.input_numeric("Min_CL", "Minimlany CL", value=1),
                                              ui.input_selectize('chose_CL','Wybierz CL do dopasowania krzywej',
                                                             [int(x) for x in range(1,20)],selected = [1,2],multiple = True),
                                              ui.input_action_button("accept_CL", "Dopasuj krzywą", class_="btn-success"),
                                              ),
                                                x.ui.accordion_panel(
                                              "Dopasowanie wariancji CL",
                                              ui.input_numeric("loss_max_var", "Maksymalna wartośc wariancji", value=100000),
                                              ui.input_numeric("Poz_CL_var", "Pozostawione wariancji", value=2),
                                              ui.input_numeric("Max_var", "Maksymalna wariancja", value=1000000),
                                              ui.input_numeric("Min_var", "Minimlana wariancja", value=0),
                                              ui.input_selectize('chose_var','Wybierz wariancje do dopasowania krzywej',
                                                              [int(x) for x in range(1,20)],selected = [1,2],multiple = True),
                                              ui.input_action_button("accept_CL_var", "Dopasuj krzywą", class_="btn-success"),
                                              ),
                                              id = 'id_panel',open=False, multiple=False),
                                          width = 2,),
            ui.panel_main(
                ui.navset_tab(
                    ui.nav("Trójkąt",
                        ui.output_table("triangle_table"),
                    ),
                    ui.nav_panel("Współczynniki CL",
                                 ui.div(
                                        ui.output_ui("ratios_table_ui"),
                                        id="panel1"
                                        )
                                 ),

                    ui.nav("Wagi",
                           ui.output_ui("binary_ratios_table_ui")
                           ),
                    ui.nav("Skumulowane CL",
                            x.ui.page_fillable(x.ui.layout_column_wrap(1, x.ui.card(ui.output_data_frame("macierz_wspol_CL_interaktywna"), ),
                                                                       height=180)),
                                    x.ui.page_fillable(x.ui.layout_column_wrap(1, x.ui.card(
                                        ui.output_data_frame("wspol_z_krzywej_CL_paid_interaktywna"), ), height=180)),
                                       x.ui.layout_column_wrap(
                                           1,
                                           x.ui.card(
                                               ui.output_plot("plot_wspolczynniki_dopasowane_interaktywny"),
                                           ),
                                           height=400),
                           ),
                    ui.nav("Wizualizacja i wyniki",
                        x.ui.page_fillable(x.ui.layout_column_wrap(1, x.ui.card(
                            ui.output_data_frame("Ult_BE_data_interaktywne"), ), height=400)),

                           ),

                ),
               ui.tags.style(css_code),
               ui.tags.script(js_code)
            )

)),


    ui.nav("Incurred Claim",
ui.layout_sidebar(ui.panel_sidebar(ui.input_selectize("linie_biznesowe_CL_incurred", "Wybierz linię biznesową", choices=['-'], multiple=False),
                                                             ui.input_numeric("ilosc_okresow_incurred", "Ilość okresów",
                                                                              value=0),
                                                             x.ui.accordion(x.ui.accordion_panel(
                                              "Dopasowanie CL",
                                              ui.input_numeric("x_incurred", "Maksymalna wartośc CL", value=3),
                                              ui.input_numeric("Poz_CL_incurred", "Pozostawione CL", value=2),
                                              ui.input_numeric("Max_CL_incurred", "Maksymalny CL", value=10),
                                              ui.input_numeric("Min_CL_incurred", "Minimlany CL", value=1),
                                              ui.input_selectize('chose_CL','Wybierz CL do dopasowania krzywej',
                                                             [int(x) for x in range(1,20)],selected = [1,2],multiple = True),
                                              ui.input_action_button("accept_CL_incurred", "Dopasuj krzywą", class_="btn-success"),
                                              ),
                                                x.ui.accordion_panel(
                                              "Dopasowanie wariancji CL",
                                              ui.input_numeric("loss_max_var_incurred", "Maksymalna wartośc wariancji", value=100000),
                                              ui.input_numeric("Poz_CL_var_incurred", "Pozostawione wariancji", value=2),
                                              ui.input_numeric("Max_var_incurred", "Maksymalna wariancja", value=1000000),
                                              ui.input_numeric("Min_var_incurred", "Minimlana wariancja", value=0),
                                              ui.input_selectize('chose_var','Wybierz wariancje do dopasowania krzywej',
                                                              [int(x) for x in range(1,20)],selected = [1,2],multiple = True),
                                              ui.input_action_button("accept_CL_var_incurred", "Dopasuj krzywą", class_="btn-success"),
                                              ),
                                              id = 'id_panel',open=False, multiple=False),
                                          width = 2,),
                  ui.panel_main(
                      ui.navset_tab(
                          ui.nav("Trójkąt",
                                 ui.output_table("triangle_table_incurred"),
                                 ),
                          ui.nav("Współczynniki CL",
                                 ui.output_ui("panel2", "ratios_table_ui_incurred")
                                 ),

                          ui.nav("Wagi",
                                 ui.output_ui("binary_ratios_table_ui_incurred")
                                 ),
                            ui.nav("Skumulowane CL",
                            x.ui.page_fillable(x.ui.layout_column_wrap(1, x.ui.card(ui.output_data_frame("macierz_wspol_CL_interaktywna_incurred"), ),
                                                                       height=180)),
                                    x.ui.page_fillable(x.ui.layout_column_wrap(1, x.ui.card(
                                        ui.output_data_frame("wspol_z_krzywej_CL_paid_interaktywna_incurred"), ), height=180)),
                                       x.ui.layout_column_wrap(
                                           1,
                                           x.ui.card(
                                               ui.output_plot("plot_wspolczynniki_dopasowane_interaktywny_incurred"),
                                           ),
                                           height=400),

                           ),
                      ),
                       # ui.tags.style(css_code),
                       # ui.tags.script(js_code)
                  )

           ),




           ),
    title="",
),

)


def server(input: Inputs, output: Outputs, session: Session):
    clicked_cells1 = reactive.Value([])
    clicked_cells2 = reactive.Value([])
    update_trigger1 = reactive.Value(0)
    update_trigger2 = reactive.Value(0)

######
    @reactive.Calc
    def triangle_paid():
        data = {
            "AY": [1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990],
            1: [5012, 106, 3410, 5655, 1092, 1513, 557, 1351, 3133, 2063],
            2: [8269, 4285, 8992, 11555, 9565, 6445, 4020, 6947, 5395, None],
            3: [10907, 5396, 13873, 15766, 15836, 11702, 10946, 13112, None, None],
            4: [11805, 10666, 16141, 21266, 22169, 12935, 12314, None, None, None],
            5: [13539, 13782, 18735, 23425, 25955, 15852, None, None, None, None],
            6: [16181, 15599, 22214, 26083, 26180, None, None, None, None, None],
            7: [18009, 15496, 22863, 27067, None, None, None, None, None, None],
            8: [18608, 16169, 23466, None, None, None, None, None, None, None],
            9: [18662, 16704, None, None, None, None, None, None, None, None],
            10: [18834, None, None, None, None, None, None, None, None, None]
        }
        df = pd.DataFrame(data)
        return df

    def triangle_incurred():
        data = {
            "AY": [1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990],
            1: [100, 106, 110, 5655, 1092, 1513, 557, 1351, 3133, 2063],
            2: [8269, 4285, 8992, 11555, 9565, 6445, 4020, 6947, 5395, None],
            3: [10907, 5396, 13873, 15766, 15836, 11702, 10946, 13112, None, None],
            4: [11805, 10666, 16141, 21266, 22169, 12935, 12314, None, None, None],
            5: [13539, 13782, 18735, 23425, 25955, 15852, None, None, None, None],
            6: [16181, 15599, 22214, 26083, 26180, None, None, None, None, None],
            7: [18009, 15496, 22863, 27067, None, None, None, None, None, None],
            8: [18608, 16169, 23466, None, None, None, None, None, None, None],
            9: [18662, 16704, None, None, None, None, None, None, None, None],
            10: [18834, None, None, None, None, None, None, None, None, None]
        }
        df = pd.DataFrame(data)
        return df

    ######
    @render.table
    def triangle_table():
        df_trian = triangle_paid()
        return df_trian

    @render.table
    def triangle_table_incurred():
        df_trian = triangle_incurred()
        return df_trian

    @reactive.Calc
    def ratio_df():
        df_input = triangle_paid()
        ratio_df_pd = yh.calculate_ratios(df_input)
        return (ratio_df_pd)

    @reactive.Calc
    def ratio_df_incurred():
        df_input = triangle_incurred()
        ratio_df_pd = yh.calculate_ratios(df_input)
        return (ratio_df_pd)

    @reactive.Calc
    def binary_df():
        ratio_df_pd = ratio_df()
        binary_df = yh.create_binary_df(ratio_df_pd)
        return (binary_df)

    @reactive.Calc
    def binary_df_incurred():
        ratio_df_pd = ratio_df_incurred()
        binary_df = yh.create_binary_df(ratio_df_pd)
        return (binary_df)

    @output
    @render.ui
    def ratios_table_ui():
        df_ratio_out = ratio_df()
        return ui.HTML(df_ratio_out.to_html(classes='table table-striped table-hover', table_id="ratios-table1"))


    @output
    @render.ui
    def ratios_table_ui_incurred():
        df_ratio_out = ratio_df_incurred()
        return ui.HTML(df_ratio_out.to_html(classes='table table-striped table-hover', table_id="ratios-table2"))

    @output
    @render.ui
    def binary_ratios_table_ui():
        # Ensure this function reacts to changes in clicked_cells
        update_trigger1.get()

        binary_df_pd = binary_df()
        return ui.HTML(binary_df_pd.to_html(classes='table table-striped table-hover', table_id="binary-ratios-table1", na_rep='NaN', float_format='{:.0f}'.format))

    @output
    @render.ui
    def binary_ratios_table_ui_incurred():
        # Ensure this function reacts to changes in clicked_cells
        binary_df_pd = binary_df_incurred()
        update_trigger2.get()
        return ui.HTML(binary_df_pd.to_html(classes='table table-striped table-hover', table_id="binary-ratios-table2",
                                            na_rep='NaN', float_format='{:.0f}'.format))

    @reactive.Effect
    @reactive.event(input.panel1_clicked_cell)
    def update_clicked_cell1():
        cell = input.panel1_clicked_cell()
        binary_df_inter = binary_df()
        if cell:
            row, col, highlighted = cell['row'], cell['col'], cell['highlighted']
            current_cells = clicked_cells1.get()
            if highlighted:
                if (row, col) not in current_cells:
                    current_cells.append((row, col))
                    binary_df_inter.iat[row, col] = 0  # Update the value to 0
            else:
                if (row, col) in current_cells:
                    current_cells.remove((row, col))
                    binary_df_inter.iat[row, col] = 1  # Update the value to 1
            print(binary_df_inter)
            clicked_cells1.set(current_cells)
            update_trigger1.set(update_trigger1.get() + 1)  # Trigger re-render

    @reactive.Effect
    @reactive.event(input.panel2_clicked_cell)
    def update_clicked_cell2():
        cell = input.panel2_clicked_cell()
        binary_df_inter = binary_df_incurred()
        if cell:
            row, col, highlighted = cell['row'], cell['col'], cell['highlighted']
            current_cells = clicked_cells2.get()
            if highlighted:
                if (row, col) not in current_cells:
                    current_cells.append((row, col))
                    binary_df_inter.iat[row, col] = 0  # Update the value to 0
            else:
                if (row, col) in current_cells:
                    current_cells.remove((row, col))
                    binary_df_inter.iat[row, col] = 1  # Update the value to 1
            clicked_cells2.set(current_cells)
            update_trigger2.set(update_trigger2.get() + 1)  # Trigger re-render

    @output
    @render.image
    def image():
        from pathlib import Path

        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "Model Ryzyka Rezerw.png"), "width": "1600px", "height": "900px"}
        return img

  #WEJSCIE
    @reactive.Calc
    def Upload_data():
        if input.file1() is None:
            return "Wprowadź dane"
        f: list[FileInfo] = input.file1()
        xl = pd.ExcelFile(f[0]["datapath"])
        sheet_names = xl.sheet_names
        #xl.close()
        return [xl, sheet_names]

    #Aktuaizacja zakladek
    @reactive.Effect
    def _():
        if input.file1() is None:
            ui.update_selectize(
                "zakladka",
                choices=['-'],
                server=True,
            )
        else:
            x = Upload_data()[1]
            ui.update_selectize(
                "zakladka",
                choices=x,
                server=False,
            )

########################################################################################################################
################################## liczenie
    #@reactive.Effect
    @reactive.event(input.clicked_cell)
    def wspolczynniki_multiplikatywna_interaktywna():
        triagnle = triangle_paid().iloc[:,1:]
        binary_df_pd = binary_df()
        binary_df_deterministic = yh.create_binary_df(triagnle)
        ind_all, m_i, m_first = yh.index_all(triagnle)
        macierz_wsp_l = yh.l_i_j(triagnle, ind_all)
        print("Dev_j_deterministic")
        Dev_j_deterministic = yh.Dev(triagnle, binary_df_deterministic, macierz_wsp_l, ind_all)
        print("Dev_j")
        Dev_j = yh.Dev(triagnle, binary_df_pd, macierz_wsp_l, ind_all)
        sigma_j = yh.sigma(triagnle, binary_df_pd, macierz_wsp_l, Dev_j, ind_all)
        sd_j = yh.wspolczynnik_sd(triagnle, binary_df_pd, sigma_j, ind_all)
        I_dataframe = pd.DataFrame(0, index=['CL_base','CL', 'sigma', 'sd'],
                                   columns=[str(x) for x in range(1, len(Dev_j) + 2)])
        I_dataframe.iloc[0, :] =  ["CL_base"]+Dev_j_deterministic
        I_dataframe.iloc[1, :] =  ["CL"]+Dev_j
        I_dataframe.iloc[2, :] =  ["sigma"]+sigma_j
        I_dataframe.iloc[3, :] =  ["sd"]+sd_j
        return I_dataframe

    #@reactive.Effect
    @reactive.event(input.clicked_cell)
    def wspolczynniki_multiplikatywna_interaktywna_incurred():
        triagnle = triangle_incurred().iloc[:,1:]
        binary_df_pd = binary_df_incurred()
        binary_df_deterministic = yh.create_binary_df(triagnle)
        ind_all, m_i, m_first = yh.index_all(triagnle)
        macierz_wsp_l = yh.l_i_j(triagnle, ind_all)
        Dev_j_deterministic = yh.Dev(triagnle, binary_df_deterministic, macierz_wsp_l, ind_all)
        Dev_j = yh.Dev(triagnle, binary_df_pd, macierz_wsp_l, ind_all)
        sigma_j = yh.sigma(triagnle, binary_df_pd, macierz_wsp_l, Dev_j, ind_all)
        sd_j = yh.wspolczynnik_sd(triagnle, binary_df_pd, sigma_j, ind_all)
        I_dataframe = pd.DataFrame(0, index=['CL_base','CL', 'sigma', 'sd'],
                                   columns=[str(x) for x in range(1, len(Dev_j) + 2)])
        I_dataframe.iloc[0, :] =  ["CL_base"]+Dev_j_deterministic
        I_dataframe.iloc[1, :] =  ["CL"]+Dev_j
        I_dataframe.iloc[2, :] =  ["sigma"]+sigma_j
        I_dataframe.iloc[3, :] =  ["sd"]+sd_j
        return I_dataframe

    @reactive.Calc
    @reactive.event(input.accept_CL, ignore_none=False)
    def dopasowanie_krzywej_factor_interaktywne():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[1,1:]
        sd_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[3,1:]
        ilosc_dop_wsp_CL = [int(x) for x in input.chose_CL()]
        vector_value, x_k_ind = yh.check_value(Dev_pd, ilosc_dop_wsp_CL,
                                               input.Min_CL(), input.Max_CL())
        sd_chose = [sd_pd[x] for x in x_k_ind]
        n_CL = len(Dev_pd) - (len(ilosc_dop_wsp_CL) - len(x_k_ind))
        a, b = yh.fit_curve(vector_value, sd_chose, x_k_ind, 'factor_CL', n_CL)
        dev_pozostawione = [Dev_pd[x] for x in range(0,(input.Poz_CL()))]
        print(dev_pozostawione)
        print(len(Dev_pd.tolist()))
        vec_output = ['CL'] + dev_pozostawione + yh.wspolczynnik_reg(a,b,input.Poz_CL() + 1,len(Dev_pd.tolist()) + input.ilosc_okresow(),'factor_CL')
        return (vec_output)

    @reactive.Calc
    @reactive.event(input.accept_CL_var, ignore_none=False)
    def dopasowanie_krzywej_variance_interaktywne():
        sigma_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[2,1:]
        sd_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[3,1:]
        ilosc_dop_wsp_var = [int(x) for x in input.chose_var()]
        sigma_pd_choose = [sigma_pd[x] for x in ilosc_dop_wsp_var]
        vector_value, x_k_ind = yh.check_value(sigma_pd_choose, ilosc_dop_wsp_var,
                                               input.Min_var(), input.Max_var())
        n_CL_var = len(sigma_pd) - (len(ilosc_dop_wsp_var) - len(x_k_ind))
        sd_chose = [sd_pd[x] for x in x_k_ind]
        a, b = yh.fit_curve(vector_value, sd_chose, x_k_ind, 'variance_CL', n_CL_var)
        sigma_pozostawione = [sigma_pd[x] for x in range(0,(input.Poz_CL_var()))]
        print(sigma_pozostawione)
        vec_output = ['sigma'] + sigma_pozostawione + yh.wspolczynnik_reg(a,b,input.Poz_CL_var() + 1,len(sigma_pd.tolist()) + input.ilosc_okresow(),'variance_CL')
        return (vec_output)

    @reactive.Calc
    def calc_chainladder_interaktywne():
        triangle = triangle_paid().iloc[:, 1:]
        CL_fit = dopasowanie_krzywej_factor_interaktywne()
        Dev_j_base = wspolczynniki_multiplikatywna_interaktywna().iloc[1,1:].to_list()
        Dev_j_z_wagami = wspolczynniki_multiplikatywna_interaktywna().iloc[1,1:].to_list()
        data_output = pd.DataFrame(0, index=np.arange(0, triangle.shape[0] + 1),
                                   columns=['Rok/Suma', 'Ult_base', 'IBNR_base', 'Ult z wagami', 'IBNR z wagami','Ult z krzywą', 'IBNR z krzywą'])
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




################# output
    @output
    @render.data_frame
    def macierz_wspol_CL_interaktywna():
        df_out_mult= wspolczynniki_multiplikatywna_interaktywna()
        return render.DataGrid(
            df_out_mult,
            width="100%",
            height="150%",
        )

    @output
    @render.data_frame
    def macierz_wspol_CL_interaktywna_incurred():
        df_out_mult= wspolczynniki_multiplikatywna_interaktywna_incurred()
        return render.DataGrid(
            df_out_mult,
            width="100%",
            height="150%",
        )

    @output
    @render.data_frame
    def wspol_z_krzywej_CL_paid_interaktywna():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[0,:]
        print("Dev_pd")
        print(Dev_pd)
        print('dopasowanie_krzywej_factor_interaktywne()')
        print(dopasowanie_krzywej_factor_interaktywne())
        size_col = len(Dev_pd)+input.ilosc_okresow()
        II_dataframe = pd.DataFrame(0,index = [0,1,2], columns = [str(x) for x in range(1,size_col+1)])
        if (len(dopasowanie_krzywej_factor_interaktywne())!=size_col and len(dopasowanie_krzywej_variance_interaktywne())==size_col):
            II_dataframe.iloc[1, :] = dopasowanie_krzywej_variance_interaktywne()
        elif (len(dopasowanie_krzywej_variance_interaktywne())!=size_col and len(dopasowanie_krzywej_factor_interaktywne())==(size_col+1)):
            II_dataframe.iloc[0, :] = dopasowanie_krzywej_factor_interaktywne()
        elif (len(dopasowanie_krzywej_variance_interaktywne()) == size_col and len(dopasowanie_krzywej_factor_interaktywne()) == size_col):
            II_dataframe.iloc[0,:] = dopasowanie_krzywej_factor_interaktywne()
            II_dataframe.iloc[1,:] = dopasowanie_krzywej_variance_interaktywne()
        else:
            II_dataframe
        print(II_dataframe.to_string())
        return render.DataGrid(
            II_dataframe,
            width="100%",
            height="100%",
            row_selection_mode='single'
        )

    @output
    @render.plot()
    def plot_wspolczynniki_dopasowane_interaktywny():
        Dev_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[0,1:]
        sigma_pd = wspolczynniki_multiplikatywna_interaktywna().iloc[1,1:]
        fig = plt.figure()
        if (input.id_panel() == None):
            plt.xticks(np.arange(1, len(sigma_pd.tolist())+input.ilosc_okresow()))
            fig.autofmt_xdate()
        elif (input.id_panel()[0]=='Dopasowanie wariancji CL'):
            var_fit = dopasowanie_krzywej_variance_interaktywne()
            plt.plot(np.arange(1,len(sigma_pd.tolist())+1), sigma_pd.to_list(), 'b',label='Sigma CL')
            plt.plot(np.arange(input.Poz_CL_var(),len(var_fit)), var_fit[input.Poz_CL_var():], 'r',label='Dopasowana Sigma CL')
            plt.xticks(np.arange(1, len(sigma_pd.tolist())+1+input.ilosc_okresow()))
            fig.autofmt_xdate()
            fig.legend()
        elif (input.id_panel()[0]=='Dopasowanie CL'):
            CL_fit = dopasowanie_krzywej_factor_interaktywne()
            plt.plot(np.arange(1,len(Dev_pd.tolist())+1), Dev_pd.to_list(), 'b',label='CL')
            plt.plot(np.arange(input.Poz_CL(),len(CL_fit)), CL_fit[input.Poz_CL():], 'r',label='Dopasowane CL')
            plt.xticks(np.arange(1, len(Dev_pd.tolist())+1+input.ilosc_okresow()))
            fig.autofmt_xdate()
            fig.legend()
        return fig

    @output
    @render.data_frame
    def Ult_BE_data_interaktywne():
        df = calc_chainladder_interaktywne()
        return render.DataGrid(
            df,
            width="100%",
            height="150%",
        )

########################################################################################################################



app = App(app_ui, server)
run_app(app)
