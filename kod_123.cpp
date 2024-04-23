#include <wx/wx.h>
#include <wx/filedlg.h>
#include <wx/splitter.h>

class MyApp : public wxApp {
public:
  virtual bool OnInit() override;
};

class MyFrame : public wxFrame {
public:
  MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);
  
private:
  void OnStartButtonClicked(wxCommandEvent& event);
  void OnIncludeAllButtonClicked(wxCommandEvent& event);
  void OnBrowseButtonClicked(wxCommandEvent& event);
  void OnLoadDataButtonClicked(wxCommandEvent& event); 
  
  wxTextCtrl* pathTextCtrl;
  wxComboBox* insurerComboBox;
  wxCheckBox* renewalCheckBox;
  wxTextCtrl* simulationsTextCtrl;
  wxTextCtrl* minSizeTextCtrl;
  wxTextCtrl* disasterDamageTextCtrl;
  wxButton* startButton;
  wxTextCtrl* outputTextCtrl; 
  
  wxDECLARE_EVENT_TABLE();
};

enum {
  ID_BUTTON_START = 10001,
  ID_BUTTON_INCLUDE_ALL = 10002,
  ID_BUTTON_BROWSE = 10003,
  ID_BUTTON_LOAD_DATA = 10004 
};

wxBEGIN_EVENT_TABLE(MyFrame, wxFrame)
  EVT_BUTTON(ID_BUTTON_START, MyFrame::OnStartButtonClicked)
  EVT_BUTTON(ID_BUTTON_INCLUDE_ALL, MyFrame::OnIncludeAllButtonClicked)
  EVT_BUTTON(ID_BUTTON_BROWSE, MyFrame::OnBrowseButtonClicked)
  EVT_BUTTON(ID_BUTTON_LOAD_DATA, MyFrame::OnLoadDataButtonClicked) 
  wxEND_EVENT_TABLE()
  
  wxIMPLEMENT_APP(MyApp);

bool MyApp::OnInit() {
  MyFrame* frame = new MyFrame("Proste GUI w wxWidgets", wxDefaultPosition, wxSize(1200, 1000));
  frame->Show(true);
  return true;
}

MyFrame::MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size)
  : wxFrame(nullptr, wxID_ANY, title, pos, size) {
  auto* splitter = new wxSplitterWindow(this, wxID_ANY);
  wxPanel* leftPanel = new wxPanel(splitter, wxID_ANY);
  wxPanel* rightPanel = new wxPanel(splitter, wxID_ANY);
  
  wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);
  leftPanel->SetSizer(vbox); 
  
  vbox->Add(new wxStaticText(leftPanel, wxID_ANY, wxT("Podaj úcieŅkÍ z danymi:")), 0, wxLEFT | wxTOP, 10);
  pathTextCtrl = new wxTextCtrl(leftPanel, wxID_ANY);
  vbox->Add(pathTextCtrl, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
  wxButton* browseButton = new wxButton(leftPanel, ID_BUTTON_BROWSE, wxT("PrzeglĻdaj..."));
  vbox->Add(browseButton, 0, wxLEFT | wxTOP, 10);
  
  vbox->Add(new wxStaticText(leftPanel, wxID_ANY, wxT("Wybierz ubezpieczyciela:")), 0, wxLEFT | wxTOP, 10);
  insurerComboBox = new wxComboBox(leftPanel, wxID_ANY);
  insurerComboBox->Append("Ubezpieczyciel A");
  insurerComboBox->Append("Ubezpieczyciel B");
  vbox->Add(insurerComboBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
  
  wxButton* includeAllButton = new wxButton(leftPanel, ID_BUTTON_INCLUDE_ALL, wxT("UwzglÍdnij wszystkich"));
  vbox->Add(includeAllButton, 0, wxLEFT | wxTOP, 10);
  
  renewalCheckBox = new wxCheckBox(leftPanel, wxID_ANY, wxT("Odnowienia"));
  vbox->Add(renewalCheckBox, 0, wxLEFT | wxTOP, 10);
  
  wxButton* loadDataButton = new wxButton(leftPanel, ID_BUTTON_LOAD_DATA, wxT("Wczytaj dane"));
  vbox->Add(loadDataButton, 0, wxLEFT | wxTOP, 10);
  
  vbox->Add(new wxStaticText(leftPanel, wxID_ANY, wxT("Iloúś symulacji:")), 0, wxLEFT | wxTOP, 10);
  simulationsTextCtrl = new wxTextCtrl(leftPanel, wxID_ANY);
  vbox->Add(simulationsTextCtrl, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
  
  vbox->Add(new wxStaticText(leftPanel, wxID_ANY, wxT("Wielkoúś minimalna:")), 0, wxLEFT | wxTOP, 10);
  minSizeTextCtrl = new wxTextCtrl(leftPanel, wxID_ANY);
  vbox->Add(minSizeTextCtrl, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
  
  vbox->Add(new wxStaticText(leftPanel, wxID_ANY, wxT("Wielkoúś szkody katastrofy:")), 0, wxLEFT | wxTOP, 10);
  disasterDamageTextCtrl = new wxTextCtrl(leftPanel, wxID_ANY);
  vbox->Add(disasterDamageTextCtrl, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
  
  startButton = new wxButton(leftPanel, ID_BUTTON_START, wxT("Start obliczeŮ"));
  vbox->Add(startButton, 0, wxLEFT | wxTOP, 10);
  
  
  
  outputTextCtrl = new wxTextCtrl(rightPanel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE | wxTE_READONLY);
  wxBoxSizer* vboxRight = new wxBoxSizer(wxVERTICAL);
  vboxRight->Add(outputTextCtrl, 1, wxEXPAND | wxALL, 5);
  rightPanel->SetSizer(vboxRight); 
  
  splitter->SplitVertically(leftPanel, rightPanel, 350); 
  splitter->SetSashGravity(0.5); 
  splitter->SetMinimumPaneSize(20); 
}

void MyFrame::OnStartButtonClicked(wxCommandEvent& event) {
  
  wxString info;
  info << wxT("ĆcieŅka z danymi: ") << pathTextCtrl->GetValue() << wxT("\n")
       << wxT("Ubezpieczyciel: ") << insurerComboBox->GetValue() << wxT("\n")
       << wxT("Odnowienia: ") << (renewalCheckBox->IsChecked() ? wxT("Tak") : wxT("Nie")) << wxT("\n")
       << wxT("Iloúś symulacji: ") << simulationsTextCtrl->GetValue() << wxT("\n")
       << wxT("Minimalna wielkoúś: ") << minSizeTextCtrl->GetValue() << wxT("\n")
       << wxT("Wielkoúś szkody katastrofy: ") << disasterDamageTextCtrl->GetValue();
  
  
  wxMessageDialog dialog(nullptr, info, wxT("Potwierdzenie danych"), wxYES_NO | wxICON_QUESTION);
  if (dialog.ShowModal() == wxID_YES) {
    
    
    int totalSteps = 100; 
    
    for (int step = 1; step <= totalSteps; step++) {
      
      wxMilliSleep(50); 
      
      
      double percentageDone = (double(step) / totalSteps) * 100;
      outputTextCtrl->SetValue(wxString::Format("Wykonano %.0f%% obliczeŮ.\n", percentageDone));
      
      wxYield(); 
    }
    
    
    outputTextCtrl->AppendText("Obliczenia zakoŮczone.\n");
  }
  else {
    
  }
}

void MyFrame::OnIncludeAllButtonClicked(wxCommandEvent& event) {
  insurerComboBox->SetValue("Wybierz wszystkich"); 
}

void MyFrame::OnBrowseButtonClicked(wxCommandEvent& event) {
  wxFileDialog openFileDialog(this, _("Wybierz plik"), "", "",
                              "Pliki tekstowe (*.txt)|*.txt|Wszystkie pliki (*.*)|*.*", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
  if (openFileDialog.ShowModal() == wxID_CANCEL)
    return;     
  
  
  pathTextCtrl->SetValue(openFileDialog.GetPath());
}

void MyFrame::OnLoadDataButtonClicked(wxCommandEvent& event) {
  wxFileDialog openFileDialog(this, _("Wybierz plik z danymi"), "", "",
                              "Pliki danych (*.dat)|*.dat|Wszystkie pliki (*.*)|*.*", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
  if (openFileDialog.ShowModal() == wxID_CANCEL) {
    return; 
  }
  
  
  wxString filePath = openFileDialog.GetPath();
  outputTextCtrl->AppendText(wxString::Format("Wczytano plik: %s\n", filePath));
}
