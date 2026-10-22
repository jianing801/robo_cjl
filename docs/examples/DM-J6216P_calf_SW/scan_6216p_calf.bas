Attribute VB_Name = "DMJ6216PCalfScan"
Option Explicit

' DM-J6216P calf assembly SolidWorks mass-property scan.
' The source contains ASCII-only text to avoid VBA editor encoding problems.
' Scan deltl = 10, 20, ..., 150 mm (15 samples), corresponding to 0.31-0.45 m.
' Mass properties use coordinate system 2, SI units, including hidden components.
' The original deltl equation is restored on success or failure. CAD documents are not saved.
'
' Before running:
' 1. Open the saved calf assembly working copy and the calf-3 length part.
' 2. Confirm the global equation is named deltl and coordinate system 2 exists.
' 3. At deltl=10 mm, verify the assembly mass properties match the screenshot values below.
'
' First-point screenshot check (coordinate system 2, deltl=10 mm):
' mass = 2.42590969 kg
' COM = (0.00019331, -0.01738978, -0.17335984) m
' principal moments = (0.00195438, 0.01303025, 0.01336786) kg*m^2

Private Const FIRST_DELTA_MM As Double = 10#
Private Const STEP_DELTA_MM As Double = 10#
Private Const SAMPLE_COUNT As Long = 15
Private Const BASE_LENGTH_M As Double = 0.3

Private Const CHECK_MASS_KG As Double = 2.42590969
Private Const CHECK_COM_X_M As Double = 0.00019331
Private Const CHECK_COM_Y_M As Double = -0.01738978
Private Const CHECK_COM_Z_M As Double = -0.17335984
Private Const CHECK_PX As Double = 0.00195438
Private Const CHECK_PY As Double = 0.01303025
Private Const CHECK_PZ As Double = 0.01336786

Private Function Num(ByVal value As Double) As String
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

Private Function FindCalfLengthPart(ByVal app As Object, _
                                    ByRef foundEquationManager As Object, _
                                    ByRef foundEquationIndex As Long) As Object
    ' Require exactly one open part whose title contains the Chinese name "calf-3"
    ' and which defines the global variable deltl.
    Dim documents As Variant
    Dim document As Object
    Dim candidateEquationManager As Object
    Dim candidateIndex As Long
    Dim candidate As Object
    Dim matches As Long
    Dim k As Long
    Dim calfPartToken As String

    Set FindCalfLengthPart = Nothing
    foundEquationIndex = -1
    calfPartToken = ChrW(&H5C0F) & ChrW(&H817F) & "3"
    documents = app.GetDocuments
    If Not IsArray(documents) Then Exit Function

    For k = LBound(documents) To UBound(documents)
        Set document = documents(k)
        If Not document Is Nothing Then
            If document.GetType = 1 Then
                If InStr(1, document.GetTitle, calfPartToken, vbTextCompare) > 0 Then
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

    If matches = 1 Then Set FindCalfLengthPart = candidate
End Function

Private Function SortedTriple(ByVal values As Variant) As Variant
    Dim result(0 To 2) As Double
    Dim i As Long
    Dim j As Long
    Dim temp As Double

    For i = 0 To 2
        result(i) = CDbl(values(i))
    Next i
    For i = 0 To 1
        For j = i + 1 To 2
            If result(j) < result(i) Then
                temp = result(i)
                result(i) = result(j)
                result(j) = temp
            End If
        Next j
    Next i
    SortedTriple = result
End Function

Private Function TensorEigenvalues(ByVal tensor As Variant, ByVal productSign As Double) As Variant
    ' Convert SolidWorks' positive-product notation to a physical inertia matrix,
    ' then compute its eigenvalues with a small Jacobi iteration.
    Dim a(0 To 2, 0 To 2) As Double
    Dim p As Long
    Dim q As Long
    Dim k As Long
    Dim iteration As Long
    Dim largest As Double
    Dim apq As Double
    Dim tau As Double
    Dim t As Double
    Dim c As Double
    Dim s As Double
    Dim akp As Double
    Dim akq As Double
    Dim eig(0 To 2) As Double

    a(0, 0) = CDbl(tensor(0))
    a(1, 1) = CDbl(tensor(4))
    a(2, 2) = CDbl(tensor(8))
    a(0, 1) = -productSign * CDbl(tensor(1)): a(1, 0) = a(0, 1)
    a(0, 2) = -productSign * CDbl(tensor(2)): a(2, 0) = a(0, 2)
    a(1, 2) = -productSign * CDbl(tensor(5)): a(2, 1) = a(1, 2)

    For iteration = 1 To 50
        largest = Abs(a(0, 1)): p = 0: q = 1
        If Abs(a(0, 2)) > largest Then largest = Abs(a(0, 2)): p = 0: q = 2
        If Abs(a(1, 2)) > largest Then largest = Abs(a(1, 2)): p = 1: q = 2
        If largest < 0.000000000000001 Then Exit For

        apq = a(p, q)
        tau = (a(q, q) - a(p, p)) / (2# * apq)
        If tau >= 0# Then
            t = 1# / (Abs(tau) + Sqr(1# + tau * tau))
        Else
            t = -1# / (Abs(tau) + Sqr(1# + tau * tau))
        End If
        c = 1# / Sqr(1# + t * t)
        s = t * c

        For k = 0 To 2
            If k <> p And k <> q Then
                akp = a(k, p): akq = a(k, q)
                a(k, p) = c * akp - s * akq: a(p, k) = a(k, p)
                a(k, q) = s * akp + c * akq: a(q, k) = a(k, q)
            End If
        Next k
        a(p, p) = a(p, p) - t * apq
        a(q, q) = a(q, q) + t * apq
        a(p, q) = 0#: a(q, p) = 0#
    Next iteration

    eig(0) = a(0, 0): eig(1) = a(1, 1): eig(2) = a(2, 2)
    TensorEigenvalues = SortedTriple(eig)
End Function

Private Function TripleError(ByVal a As Variant, ByVal b As Variant) As Double
    Dim i As Long
    Dim difference As Double
    TripleError = 0#
    For i = 0 To 2
        difference = Abs(CDbl(a(i)) - CDbl(b(i)))
        If difference > TripleError Then TripleError = difference
    Next i
End Function

Private Function DetectProductSign(ByVal tensor As Variant, ByVal principalMoments As Variant) As Double
    Dim target As Variant
    Dim eigPositive As Variant
    Dim eigNegative As Variant
    Dim errorPositive As Double
    Dim errorNegative As Double

    target = SortedTriple(principalMoments)
    eigPositive = TensorEigenvalues(tensor, 1#)
    eigNegative = TensorEigenvalues(tensor, -1#)
    errorPositive = TripleError(eigPositive, target)
    errorNegative = TripleError(eigNegative, target)

    If errorPositive < errorNegative Then
        DetectProductSign = 1#
    Else
        DetectProductSign = -1#
    End If
    If errorPositive > 0.00000001 And errorNegative > 0.00000001 Then
        Err.Raise 2040, , "The inertia-tensor eigenvalues do not match the SolidWorks principal moments. Scan stopped."
    End If
End Function

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
    Dim calfToken As String
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
    Dim productSign As Double
    Dim rebuildOK As Boolean

    Dim center As Variant
    Dim inertiaAtCenter As Variant
    Dim inertiaAtOrigin As Variant
    Dim principalMoments As Variant
    Dim sortedMoments As Variant
    Dim axisX As Variant
    Dim axisY As Variant
    Dim axisZ As Variant

    Set app = Application.SldWorks
    Set assembly = app.ActiveDoc
    If assembly Is Nothing Then Err.Raise 2001, , "Activate the saved calf assembly working copy before running this macro."
    If assembly.GetType <> 2 Then Err.Raise 2002, , "The active document is not an assembly."
    calfToken = ChrW(&H5C0F) & ChrW(&H817F)
    If InStr(1, assembly.GetTitle, calfToken, vbTextCompare) = 0 Then
        Err.Raise 2003, , "The active document title does not identify a calf assembly. Scan stopped."
    End If

    assemblyPath = assembly.GetPathName
    If Len(assemblyPath) = 0 Then Err.Raise 2004, , "Save the assembly working copy first so the CSV folder can be located."
    outputDirectory = Left$(assemblyPath, InStrRev(assemblyPath, "\") - 1)

    Set part = FindCalfLengthPart(app, equationManager, equationIndex)
    If part Is Nothing Then
        Err.Raise 2005, , "Could not uniquely find the open calf-3 length part with global equation deltl."
    End If

    originalEquation = equationManager.Equation(equationIndex)
    coordinateName = ChrW(&H5750) & ChrW(&H6807) & ChrW(&H7CFB) & "2"
    outputFile = outputDirectory & "\6216P_calf_SW_scan_" & Format$(Now, "yyyymmdd_hhnnss") & ".csv"
    statusFile = outputDirectory & "\6216P_calf_scan_status_" & Format$(Now, "yyyymmdd_hhnnss") & ".txt"

    csvHeader = "record_type,length_m,nominal_length_m,delta_mm,mass_kg,volume_m3,surface_area_m2,com_x_m,com_y_m,com_z_m"
    csvHeader = csvHeader & ",Px,Py,Pz,axis_ix_x,axis_ix_y,axis_ix_z,axis_iy_x,axis_iy_y,axis_iy_z,axis_iz_x,axis_iz_y,axis_iz_z"
    csvHeader = csvHeader & ",Lxx,Lxy,Lxz,Lyx,Lyy,Lyz,Lzx,Lzy,Lzz,Ixx,Ixy,Ixz,Iyx,Iyy,Iyz,Izx,Izy,Izz"
    csvHeader = csvHeader & ",api_to_SW_product_sign,coordinate_system,part_rebuild_ok,assembly_rebuild_ok"

    fileHandle = FreeFile
    Open outputFile For Output As #fileHandle
    Print #fileHandle, csvHeader

    For sample = 1 To SAMPLE_COUNT
        deltaMm = FIRST_DELTA_MM + (sample - 1) * STEP_DELTA_MM
        nominalLengthM = BASE_LENGTH_M + deltaMm / 1000#
        actualLengthM = nominalLengthM

        Set activeDocument = app.ActivateDoc3(part.GetTitle, False, 2, activateErrors)
        If activeDocument Is Nothing Then Err.Raise 2010, , "Cannot activate the calf-3 length part."
        equationManager.Equation(equationIndex) = """deltl"" = " & Num(deltaMm) & "mm"
        evaluateErrors = equationManager.EvaluateAll
        rebuildOK = part.ForceRebuild3(False)
        If Not rebuildOK Then Err.Raise 2011, , "Length-part rebuild failed at deltl=" & Num(deltaMm) & " mm."

        Set activeDocument = app.ActivateDoc3(assembly.GetTitle, False, 2, activateErrors)
        If activeDocument Is Nothing Then Err.Raise 2012, , "Cannot activate the calf assembly."
        rebuildOK = assembly.ForceRebuild3(False)
        If Not rebuildOK Then Err.Raise 2013, , "Assembly rebuild failed at deltl=" & Num(deltaMm) & " mm."

        assembly.ClearSelection2 True
        Set coordinateTransform = assembly.Extension.GetCoordinateSystemTransformByName(coordinateName)
        If coordinateTransform Is Nothing Then Err.Raise 2014, , "Coordinate system 2 was not found in the assembly."

        Set massProperty = assembly.Extension.CreateMassProperty2
        If massProperty Is Nothing Then Err.Raise 2015, , "Cannot create the mass-property object."
        massProperty.UseSystemUnits = True
        massProperty.IncludeHiddenBodiesOrComponents = True
        If Not massProperty.SetCoordinateSystem(coordinateTransform) Then Err.Raise 2016, , "Cannot set mass properties to coordinate system 2."
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
                Err.Raise 2020, , "The 10-mm mass does not match the supplied coordinate-system-2 screenshot. Check motor overrides and configuration."
            End If
            If Abs(CDbl(center(0)) - CHECK_COM_X_M) > 0.00005 Or _
               Abs(CDbl(center(1)) - CHECK_COM_Y_M) > 0.00005 Or _
               Abs(CDbl(center(2)) - CHECK_COM_Z_M) > 0.00005 Then
                Err.Raise 2021, , "The 10-mm COM does not match the supplied coordinate-system-2 screenshot."
            End If
            sortedMoments = SortedTriple(principalMoments)
            If Abs(CDbl(sortedMoments(0)) - CHECK_PX) > 0.000005 Or _
               Abs(CDbl(sortedMoments(1)) - CHECK_PY) > 0.000005 Or _
               Abs(CDbl(sortedMoments(2)) - CHECK_PZ) > 0.000005 Then
                Err.Raise 2022, , "The 10-mm principal moments do not match the supplied screenshot."
            End If
            productSign = DetectProductSign(inertiaAtCenter, principalMoments)
        End If

        For k = 0 To 8
            If k <> 0 And k <> 4 And k <> 8 Then
                inertiaAtCenter(k) = inertiaAtCenter(k) * productSign
                inertiaAtOrigin(k) = inertiaAtOrigin(k) * productSign
            End If
        Next k

        rowText = "sample," & Num(actualLengthM) & "," & Num(nominalLengthM) & "," & Num(deltaMm)
        rowText = rowText & "," & Num(massProperty.Mass) & "," & Num(massProperty.Volume) & "," & Num(massProperty.SurfaceArea)
        rowText = AddArray(rowText, center)
        rowText = AddArray(rowText, principalMoments)
        rowText = AddArray(rowText, axisX)
        rowText = AddArray(rowText, axisY)
        rowText = AddArray(rowText, axisZ)
        rowText = AddArray(rowText, inertiaAtCenter)
        rowText = AddArray(rowText, inertiaAtOrigin)
        rowText = rowText & "," & Num(productSign) & ",coordinate_system_2,True,True"
        Print #fileHandle, rowText
        rowsWritten = rowsWritten + 1
        DoEvents
    Next sample

    Close #fileHandle
    fileHandle = 0

    ' Restore the exact equation string that was present before the scan.
    Set activeDocument = app.ActivateDoc3(part.GetTitle, False, 2, activateErrors)
    equationManager.Equation(equationIndex) = originalEquation
    evaluateErrors = equationManager.EvaluateAll
    rebuildOK = part.ForceRebuild3(False)
    If Not rebuildOK Then Err.Raise 2030, , "Scan finished, but restoring the original deltl equation failed to rebuild."
    Set activeDocument = app.ActivateDoc3(assembly.GetTitle, False, 2, activateErrors)
    rebuildOK = assembly.ForceRebuild3(False)
    If Not rebuildOK Then Err.Raise 2031, , "Scan finished, but the restored assembly failed to rebuild."

    statusHandle = FreeFile
    Open statusFile For Output As #statusHandle
    Print #statusHandle, "SUCCESS: " & rowsWritten & " samples; deltl=10..150 mm, step 10 mm"
    Print #statusHandle, "Measured leg lengths: 0.31..0.45 m; design/prediction range: 0.30..0.45 m"
    Print #statusHandle, outputFile
    Print #statusHandle, "Coordinate system: coordinate_system_2"
    Print #statusHandle, "Restored original equation: " & originalEquation
    Print #statusHandle, "No CAD files saved by the macro."
    Close #statusHandle

    MsgBox "6216P calf scan complete: " & rowsWritten & " samples (0.31 to 0.45 m), using coordinate system 2." & vbCrLf & _
           "The original equation was restored. No CAD files were saved." & vbCrLf & outputFile, vbInformation
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
        Print #statusHandle, "The macro attempted to restore the original equation. No CAD files were saved."
        Close #statusHandle
    End If
    MsgBox failureText, vbExclamation
End Sub
