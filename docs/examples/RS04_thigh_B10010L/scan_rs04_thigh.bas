Attribute VB_Name = "RS04ThighScan"
Option Explicit

' RS04 thigh assembly scan; coordinate system 1.
' Scan deltl=10,20,...,150 mm: 15 points, lengths 0.26 to 0.40 m.
' SI units, eight decimals, include hidden bodies and components.
' Open the thigh-3 length part, then activate the saved assembly copy.
' Restore the original equation after scanning. No CAD files are saved.
' Chinese object names are constructed with ChrW to avoid encoding issues.
' See the separate Chinese usage guide for the import/run instructions.

Private Const FIRST_DELTA_MM As Double = 10#
Private Const STEP_DELTA_MM As Double = 10#
Private Const SAMPLE_COUNT As Long = 15
Private Const BASE_LENGTH_M As Double = 0.25
Private Const CHECK_MASS_KG As Double = 2.29882456
Private Const CHECK_LYZ As Double = 0.00127222
Private Const CHECK_COM_X As Double = 0.00064559
Private Const CHECK_COM_Y As Double = -0.01227559
Private Const CHECK_COM_Z As Double = -0.21356263
Private Const CHECK_LXX As Double = 0.01830738
Private Const CHECK_LYY As Double = 0.02000587
Private Const CHECK_LZZ As Double = 0.00302357

Private Function Num(ByVal value As Double) As String
    ' Export eight decimals with a dot as the decimal separator.
    Num = Replace(Format$(value, "0.00000000"), ",", ".")
End Function

Private Function AddArray(ByVal rowText As String, ByVal values As Variant) As String
    Dim k As Long
    For k = LBound(values) To UBound(values)
        rowText = rowText & "," & Num(CDbl(values(k)))
    Next k
    AddArray = rowText
End Function

Private Function FindDeltlEquation(ByVal part As Object, ByRef equationManager As Object) As Long
    Dim k As Long
    Dim equationText As String
    Dim leftSide As String
    Dim equalAt As Long

    FindDeltlEquation = -1
    Set equationManager = part.GetEquationMgr
    If equationManager Is Nothing Then Exit Function

    For k = 0 To equationManager.GetCount - 1
        equationText = Trim$(equationManager.Equation(k))
        equalAt = InStr(1, equationText, "=", vbTextCompare)
        If equalAt > 0 Then
            leftSide = Trim$(Left$(equationText, equalAt - 1))
            If LCase$(leftSide) = """deltl""" Then
                FindDeltlEquation = k
                Exit Function
            End If
        End If
    Next k
End Function

Private Function FindLengthPart(ByVal app As Object, ByRef foundEquationManager As Object, ByRef foundEquationIndex As Long) As Object
    ' Find the opened thigh-3 part by its Unicode title and deltl variable.
    ' Exactly one matching length part must be open.
    Dim documents As Variant
    Dim document As Object
    Dim candidateEquationManager As Object
    Dim candidateIndex As Long
    Dim candidate As Object
    Dim matches As Long
    Dim k As Long

    Set FindLengthPart = Nothing
    foundEquationIndex = -1
    documents = app.GetDocuments
    If Not IsArray(documents) Then Exit Function

    For k = LBound(documents) To UBound(documents)
        Set document = documents(k)
        If Not document Is Nothing Then
            If document.GetType = 1 Then
                If InStr(1, document.GetTitle, ChrW(&H5927) & ChrW(&H817F) & "3", vbTextCompare) > 0 Then
                    candidateIndex = FindDeltlEquation(document, candidateEquationManager)
                    If candidateIndex >= 0 Then
                        matches = matches + 1
                        Set candidate = document
                        Set foundEquationManager = candidateEquationManager
                        foundEquationIndex = candidateIndex
                    End If
                End If
            End If
        End If
    Next k

    If matches = 1 Then Set FindLengthPart = candidate
End Function

Private Function AssemblyHasMotorPlaceholder(ByVal assembly As Object) As Boolean
    ' The screenshot uses dm-j10010l geometry with RS04 mass overrides.
    Dim components As Variant
    Dim component As Object
    Dim componentName As String
    Dim componentPath As String
    Dim k As Long

    AssemblyHasMotorPlaceholder = False
    components = assembly.GetComponents(True)
    If Not IsArray(components) Then Exit Function

    For k = LBound(components) To UBound(components)
        Set component = components(k)
        If Not component Is Nothing Then
            componentName = LCase$(component.Name2)
            componentPath = LCase$(component.GetPathName)
            If InStr(1, componentName, "10010l", vbTextCompare) > 0 Or _
               InStr(1, componentPath, "10010l", vbTextCompare) > 0 Then
                AssemblyHasMotorPlaceholder = True
                Exit Function
            End If
        End If
    Next k
End Function

Private Sub CheckParallelAxis(ByVal mass As Double, ByVal center As Variant, _
                              ByVal central As Variant, ByVal origin As Variant)
    ' SW positive products: P_origin = P_COM + m * c_i * c_j.
    ' Diagonal: I_origin_xx = I_COM_xx + m * (cy^2 + cz^2).
    Dim cx As Double, cy As Double, cz As Double
    Dim tolerance As Double
    cx = CDbl(center(0)): cy = CDbl(center(1)): cz = CDbl(center(2))
    tolerance = 0.0000001
    If Abs(CDbl(origin(0)) - CDbl(central(0)) - mass * (cy * cy + cz * cz)) > tolerance Or _
       Abs(CDbl(origin(4)) - CDbl(central(4)) - mass * (cx * cx + cz * cz)) > tolerance Or _
       Abs(CDbl(origin(8)) - CDbl(central(8)) - mass * (cx * cx + cy * cy)) > tolerance Then
        Err.Raise 2024, , "Parallel-axis check failed for origin and COM inertia. Scan stopped."
    End If
    If Abs(CDbl(origin(1)) - CDbl(central(1)) - mass * cx * cy) > tolerance Or _
       Abs(CDbl(origin(2)) - CDbl(central(2)) - mass * cx * cz) > tolerance Or _
       Abs(CDbl(origin(5)) - CDbl(central(5)) - mass * cy * cz) > tolerance Then
        Err.Raise 2025, , "Product-of-inertia sign or coordinate-system check failed."
    End If
End Sub

Public Sub main()
    On Error GoTo Failed

    Dim app As Object
    Dim assembly As Object
    Dim part As Object
    Dim equationManager As Object
    Dim massProperty As Object
    Dim coordinateTransform As Object
    Dim activeDocument As Object

    Dim assemblyPath As String
    Dim outputDirectory As String
    Dim outputFile As String
    Dim statusFile As String
    Dim coordinateName As String
    Dim originalEquation As String
    Dim csvHeader As String
    Dim rowText As String
    Dim failureText As String

    Dim equationIndex As Long
    Dim activateErrors As Long
    Dim evaluateErrors As Long
    Dim fileHandle As Integer
    Dim statusHandle As Integer
    Dim rowsWritten As Long
    Dim sample As Long
    Dim k As Long
    Dim deltaMm As Double
    Dim nominalLengthM As Double
    Dim actualLengthM As Double
    Dim previousMass As Double
    Dim productSign As Double
    Dim signTolerance As Double
    Dim rebuildOK As Boolean

    Dim center As Variant
    Dim inertiaAtCenter As Variant
    Dim inertiaAtOrigin As Variant
    Dim principalMoments As Variant
    Dim axisX As Variant
    Dim axisY As Variant
    Dim axisZ As Variant

    Set app = Application.SldWorks
    Set assembly = app.ActiveDoc
    If assembly Is Nothing Then Err.Raise 2001, , "Activate the RS04 thigh assembly before running this macro."
    If assembly.GetType <> 2 Then Err.Raise 2002, , "The active document is not an assembly."
    If Not AssemblyHasMotorPlaceholder(assembly) Then Err.Raise 2005, , "The expected dm-j10010l motor placeholder was not found."

    assemblyPath = assembly.GetPathName
    If Len(assemblyPath) = 0 Then Err.Raise 2003, , "Save the assembly working copy first to define the CSV output folder."
    outputDirectory = Left$(assemblyPath, InStrRev(assemblyPath, "\") - 1)

    Set part = FindLengthPart(app, equationManager, equationIndex)
    If part Is Nothing Then
        Err.Raise 2004, , "Open exactly one thigh-3 length part with the global variable deltl, then activate the assembly."
    End If

    originalEquation = equationManager.Equation(equationIndex)
    coordinateName = ChrW(&H5750) & ChrW(&H6807) & ChrW(&H7CFB) & "1"
    outputFile = outputDirectory & "\RS04_thigh_SW_scan_" & Format$(Now, "yyyymmdd_hhnnss") & ".csv"
    statusFile = outputDirectory & "\RS04_thigh_scan_status_" & Format$(Now, "yyyymmdd_hhnnss") & ".txt"

    csvHeader = "record_type,length_m,nominal_length_m,delta_mm,mass_kg,volume_m3,surface_area_m2,com_x_m,com_y_m,com_z_m"
    csvHeader = csvHeader & ",Px,Py,Pz,axis_ix_x,axis_ix_y,axis_ix_z,axis_iy_x,axis_iy_y,axis_iy_z,axis_iz_x,axis_iz_y,axis_iz_z"
    csvHeader = csvHeader & ",Lxx,Lxy,Lxz,Lyx,Lyy,Lyz,Lzx,Lzy,Lzz,Ixx,Ixy,Ixz,Iyx,Iyy,Iyz,Izx,Izy,Izz"
    csvHeader = csvHeader & ",api_to_SW_product_sign,coordinate_system,part_rebuild_ok,assembly_rebuild_ok"

    fileHandle = FreeFile
    Open outputFile For Output As #fileHandle
    Print #fileHandle, csvHeader

    signTolerance = 0.00000005
    For sample = 1 To SAMPLE_COUNT
        deltaMm = FIRST_DELTA_MM + (sample - 1) * STEP_DELTA_MM
        nominalLengthM = BASE_LENGTH_M + deltaMm / 1000#
        actualLengthM = nominalLengthM

        Set activeDocument = app.ActivateDoc3(part.GetTitle, False, 2, activateErrors)
        If activeDocument Is Nothing Then Err.Raise 2010, , "Cannot activate the thigh-3 length part."

        equationManager.Equation(equationIndex) = """deltl"" = " & Num(deltaMm) & "mm"
        evaluateErrors = equationManager.EvaluateAll
        rebuildOK = part.ForceRebuild3(False)
        If Not rebuildOK Then Err.Raise 2011, , "Length-part rebuild failed; deltl=" & Num(deltaMm) & " mm."

        Set activeDocument = app.ActivateDoc3(assembly.GetTitle, False, 2, activateErrors)
        If activeDocument Is Nothing Then Err.Raise 2012, , "Cannot activate the thigh assembly."
        rebuildOK = assembly.ForceRebuild3(False)
        If Not rebuildOK Then Err.Raise 2013, , "Assembly rebuild failed; deltl=" & Num(deltaMm) & " mm."

        assembly.ClearSelection2 True
        Set coordinateTransform = assembly.Extension.GetCoordinateSystemTransformByName(coordinateName)
        If coordinateTransform Is Nothing Then Err.Raise 2014, , "Coordinate system 1 was not found in the assembly."

        Set massProperty = assembly.Extension.CreateMassProperty2
        If massProperty Is Nothing Then Err.Raise 2015, , "Cannot create the mass-property object."
        massProperty.UseSystemUnits = True
        massProperty.IncludeHiddenBodiesOrComponents = True
        If Not massProperty.SetCoordinateSystem(coordinateTransform) Then Err.Raise 2016, , "Cannot set mass properties to coordinate system 1."
        massProperty.Recalculate

        center = massProperty.CenterOfMass
        inertiaAtCenter = massProperty.GetMomentOfInertia(0)
        inertiaAtOrigin = massProperty.GetMomentOfInertia(1)
        principalMoments = massProperty.PrincipalMomentsOfInertia
        axisX = massProperty.PrincipalAxesOfInertia(0)
        axisY = massProperty.PrincipalAxesOfInertia(1)
        axisZ = massProperty.PrincipalAxesOfInertia(2)

        If sample = 1 Then
            If Abs(massProperty.Mass - CHECK_MASS_KG) > 0.00005 Then
                Err.Raise 2020, , "The 10-mm mass does not match the RS04 screenshot. Check motor overrides and configuration."
            End If
            If Abs(CDbl(center(0)) - CHECK_COM_X) > 0.000002 Or _
               Abs(CDbl(center(1)) - CHECK_COM_Y) > 0.000002 Or _
               Abs(CDbl(center(2)) - CHECK_COM_Z) > 0.000002 Then
                Err.Raise 2022, , "The 10-mm COM does not match the coordinate-system-1 screenshot."
            End If
            If Abs(CDbl(inertiaAtCenter(0)) - CHECK_LXX) > signTolerance Or _
               Abs(CDbl(inertiaAtCenter(4)) - CHECK_LYY) > signTolerance Or _
               Abs(CDbl(inertiaAtCenter(8)) - CHECK_LZZ) > signTolerance Then
                Err.Raise 2023, , "The 10-mm COM inertia diagonals do not match the RS04 screenshot."
            End If
            If Abs(inertiaAtCenter(5) - CHECK_LYZ) < signTolerance Then
                productSign = 1#
            ElseIf Abs(inertiaAtCenter(5) + CHECK_LYZ) < signTolerance Then
                productSign = -1#
            Else
                Err.Raise 2021, , "The 10-mm Lyz does not match the screenshot; product sign cannot be calibrated."
            End If
        End If

        ' Calibrate the API product-of-inertia sign from the first screenshot point.
        For k = 0 To 8
            If k <> 0 And k <> 4 And k <> 8 Then
                inertiaAtCenter(k) = inertiaAtCenter(k) * productSign
                inertiaAtOrigin(k) = inertiaAtOrigin(k) * productSign
            End If
        Next k

        Call CheckParallelAxis(CDbl(massProperty.Mass), center, inertiaAtCenter, inertiaAtOrigin)
        If sample > 1 Then
            If CDbl(massProperty.Mass) <= previousMass + 0.0000000001 Then
                Err.Raise 2026, , "Assembly mass did not increase with length. Check the length-part update."
            End If
        End If
        previousMass = CDbl(massProperty.Mass)

        rowText = "sample," & Num(actualLengthM) & "," & Num(nominalLengthM) & "," & Num(deltaMm)
        rowText = rowText & "," & Num(massProperty.Mass) & "," & Num(massProperty.Volume) & "," & Num(massProperty.SurfaceArea)
        rowText = AddArray(rowText, center)
        rowText = AddArray(rowText, principalMoments)
        rowText = AddArray(rowText, axisX)
        rowText = AddArray(rowText, axisY)
        rowText = AddArray(rowText, axisZ)
        rowText = AddArray(rowText, inertiaAtCenter)
        rowText = AddArray(rowText, inertiaAtOrigin)
        rowText = rowText & "," & Num(productSign) & ",coordinate_system_1,True,True"
        Print #fileHandle, rowText
        rowsWritten = rowsWritten + 1
        DoEvents
    Next sample

    Close #fileHandle
    fileHandle = 0

    ' Restore the original equation and rebuild without saving CAD files.
    Set activeDocument = app.ActivateDoc3(part.GetTitle, False, 2, activateErrors)
    equationManager.Equation(equationIndex) = originalEquation
    evaluateErrors = equationManager.EvaluateAll
    rebuildOK = part.ForceRebuild3(False)
    If Not rebuildOK Then Err.Raise 2030, , "Scan finished, but restoring the original length-part equation failed to rebuild."

    Set activeDocument = app.ActivateDoc3(assembly.GetTitle, False, 2, activateErrors)
    rebuildOK = assembly.ForceRebuild3(False)
    If Not rebuildOK Then Err.Raise 2031, , "Scan finished, but the restored assembly failed to rebuild."

    statusHandle = FreeFile
    Open statusFile For Output As #statusHandle
    Print #statusHandle, "SUCCESS: " & rowsWritten & " sample rows (10 to 150 mm, step 10 mm)"
    Print #statusHandle, outputFile
    Print #statusHandle, "Restored original equation: " & originalEquation
    Print #statusHandle, "No CAD files saved by the macro."
    Close #statusHandle

    MsgBox "RS04 scan complete: " & rowsWritten & " samples (0.26 to 0.40 m)." & vbCrLf & _
           "Original equation restored. No CAD files were saved." & vbCrLf & outputFile, vbInformation
    Exit Sub

Failed:
    failureText = "ERROR " & Err.Number & ": " & Err.Description & "; rows=" & rowsWritten
    On Error Resume Next
    If fileHandle <> 0 Then Close #fileHandle

    If Not equationManager Is Nothing And equationIndex >= 0 And Len(originalEquation) > 0 Then
        Set activeDocument = app.ActivateDoc3(part.GetTitle, False, 2, activateErrors)
        equationManager.Equation(equationIndex) = originalEquation
        evaluateErrors = equationManager.EvaluateAll
        rebuildOK = part.ForceRebuild3(False)
        Set activeDocument = app.ActivateDoc3(assembly.GetTitle, False, 2, activateErrors)
        rebuildOK = assembly.ForceRebuild3(False)
    End If

    If Len(statusFile) > 0 Then
        statusHandle = FreeFile
        Open statusFile For Output As #statusHandle
        Print #statusHandle, failureText
        If Len(outputFile) > 0 Then Print #statusHandle, outputFile
        Print #statusHandle, "The macro attempted to restore the original deltl equation. No CAD files were saved."
        Close #statusHandle
    End If
    MsgBox failureText, vbExclamation
End Sub
