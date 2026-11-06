; Official Inno Setup 6 Installer Script for NEVER-AI (Windows 10/11 x64)
; Produces a standard, Defender-clean Windows Setup Wizard (NeverAI-Setup-x64.exe)
; with zero PowerShell scripts, zero batch files, and zero UPX packing.

#define MyAppName "NEVER-AI"
#define MyAppVersion "1.4.1"
#define MyAppPublisher "Never-AI Open Research"
#define MyAppURL "https://github.com/mouusieur/never-ai"
#define MyAppExeName "NeverAI.exe"

[Setup]
AppId={{8E4F7B2A-91C4-4D3E-A612-7F98E104C2B1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/releases
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\NeverAI
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\out_installer
OutputBaseFilename=NeverAI-Setup-x64
SetupIconFile=..\..\assets\brand\never_ai.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
DisableProgramGroupPage=yes
VersionInfoVersion=1.4.1.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=NEVER-AI Sovereign Desktop Installer (Windows x64)
VersionInfoProductName={#MyAppName}

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\NeverAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\never_ai_data"
Name: "{app}\never_ai_data\checkpoints"
Name: "{app}\never_ai_data\memory_store"
Name: "{app}\never_ai_data\logs"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
