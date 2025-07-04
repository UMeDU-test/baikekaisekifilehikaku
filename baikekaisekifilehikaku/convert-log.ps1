# PowerShell 2.0 完全対応版 (高速化 ＆ 進捗表示機能付き)

# --- 設定項目はありません。スクリプトを実行するとファイル選択画面が表示されます。 ---

# GUIのファイル選択ダイアログを使用するために必要なアセンブリを読み込む
try {
    Add-Type -AssemblyName System.Windows.Forms
}
catch {
    Write-Warning "GUIコンポーネントの読み込みに失敗しました。スクリプトを続行できません。"
    if ($Host.Name -eq "ConsoleHost") { Read-Host "何かキーを押して終了してください" }
    return
}

# ファイル選択ダイアログのオブジェクトを作成
$openFileDialog = New-Object System.Windows.Forms.OpenFileDialog
$openFileDialog.Title = "変換するCSVファイルを選択してください"
$openFileDialog.Filter = "CSVファイル (*.csv)|*.csv|すべてのファイル (*.*)|*.*"
$openFileDialog.InitialDirectory = [Environment]::GetFolderPath('Desktop') 

# ダイアログを表示し、ユーザーがファイルを選択するのを待つ
if ($openFileDialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    $inputCsvPath = $openFileDialog.FileName
}
else {
    Write-Host "ファイルが選択されなかったため、処理を中断しました。"
    if ($Host.Name -eq "ConsoleHost") { Read-Host "何かキーを押して終了してください" }
    return
}

# --- 出力ファイルパスの自動生成 ---
$inputDirectory = Split-Path -Path $inputCsvPath -Parent
$inputBaseName  = [System.IO.Path]::GetFileNameWithoutExtension($inputCsvPath)
$outputCsvPath  = Join-Path -Path $inputDirectory -ChildPath "${inputBaseName}_converted.csv"

# --- ここから下はデータ変換処理です ---

Write-Host "----------------------------------------"
Write-Host "処理を開始します..."
Write-Host "入力ファイル: $inputCsvPath"
Write-Host "----------------------------------------"

try {
    # ★★★ 高速化のための変更点 (ここから) ★★★

    # 1. 新しいヘッダーとマッピングを定義
    $newHeaders = @("Time","Lap","Latitude","Longitude","Speed_kmh","PBKF","PBKR","GYXAC","GYPIAN","Throttle_pct","Lean_Angle","Gear_Position","RPM","Wheel_Speed_F","Wheel_Speed_R","AFR","TC_Intervention","TRQPCTRD","SLIP")
    $oldHeaderMap = @{'Time'='Time';'Lap'='Lap';'Latitude'='GPS lat';'Longitude'='GPS lon';'RPM'='Ne';'Throttle_pct'='TPS';'Speed_kmh'='GPSspd';'Wheel_Speed_F'='VFW';'Wheel_Speed_R'='VRW';'Gear_Position'='GearPos';'Lean_Angle'='LeanAng';'AFR'='LAF';'SLIP'='SLIP';'PBKF'='PBKF';'PBKR'='PBKR';'GYPIAN'='GYPIAN';'GYXAC'='GYXAC';'TC_Intervention'='STRQTRG';'TRQPCTRD'='TRQPCTRD'}

    # 2. .NETのStreamReaderでファイルを一行ずつ効率的に読み込む
    $reader = New-Object System.IO.StreamReader($inputCsvPath)

    # 3. 最初の6行を読み飛ばす
    Write-Host "1. ヘッダー情報を解析しています..."
    for ($i = 0; $i -lt 6; $i++) { $null = $reader.ReadLine() }

    # 4. 7行目のヘッダー行を読み込み、列の対応表（インデックス）を作成する
    $oldHeaderLine = $reader.ReadLine()
    $oldHeaders = $oldHeaderLine.Split(',')
    $columnIndexMap = @{}
    for ($i = 0; $i -lt $newHeaders.Count; $i++) {
        $oldHeaderName = $oldHeaderMap[$newHeaders[$i]]
        if ($oldHeaderName) {
            $oldIndex = [array]::IndexOf($oldHeaders, $oldHeaderName)
            if ($oldIndex -ge 0) {
                $columnIndexMap[$i] = $oldIndex
            }
        }
    }
    Write-Host "   ヘッダー解析完了。"

    # 5. 出力用の文字列を効率的に作成するStringBuilderを準備
    $stringBuilder = New-Object System.Text.StringBuilder
    # 新しいヘッダー行を書き込む
    $null = $stringBuilder.AppendLine(($newHeaders -join ','))

    # 6. データ行を一行ずつ処理
    Write-Host "2. データを変換・書き出し準備をしています..."
    $lineCounter = 0
    while (($line = $reader.ReadLine()) -ne $null) {
        $lineCounter++
        $oldValues = $line.Split(',')
        $newValues = New-Object string[] $newHeaders.Count

        # 作成した対応表を元に、新しい行のデータを作成
        foreach ($newIndex in $columnIndexMap.Keys) {
            $oldIndex = $columnIndexMap[$newIndex]
            if ($oldIndex -lt $oldValues.Count) {
                $newValues[$newIndex] = $oldValues[$oldIndex]
            }
        }
        $null = $stringBuilder.AppendLine(($newValues -join ','))
        
        # 1000行ごとに進捗を表示
        if ($lineCounter % 1000 -eq 0) {
            Write-Progress -Activity "データ変換中" -Status "$lineCounter 行を処理済み" -Id 1
        }
    }
    $reader.Close()
    Write-Progress -Activity "データ変換中" -Completed -Id 1
    Write-Host "   データ変換完了。"
    
    # 7. 構築した全文字列を一度にファイルに書き出す
    Write-Host "3. 新しいCSVファイルに書き出しています..."
    [System.IO.File]::WriteAllText($outputCsvPath, $stringBuilder.ToString(), [System.Text.Encoding]::UTF8)

    # ★★★ 高速化のための変更点 (ここまで) ★★★
    
    Write-Host "   書き出し完了。"
    Write-Host "----------------------------------------"
    Write-Host "すべての処理が完了しました！" -ForegroundColor Green
    Write-Host "出力ファイル: $outputCsvPath"
}
catch {
    Write-Host ""
    Write-Host "★★エラー★★" -ForegroundColor Red
    Write-Host "処理中にエラーが発生しました。" -ForegroundColor Red
    Write-Host "エラー詳細: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    if($reader) { $reader.Dispose() }
    if ($Host.Name -eq "ConsoleHost") { Read-Host "何かキーを押してウィンドウを閉じてください" }
}