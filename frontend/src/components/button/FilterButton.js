import "./styles/main.sass"

export function FilterButton({
    onClick,
    children = undefined,
    className = "",
}){
    return(
        <button
            className={`filter_detail__button ${className}`}
            onClick={onClick}
        >
            {children}
        </button>
    )
}